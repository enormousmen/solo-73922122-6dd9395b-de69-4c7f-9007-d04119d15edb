"""备用卷分层副本引擎。

把源卷某个恢复点及其全部祖先层，按原父子关系复制到备用卷（REPLICA_DIR，
容器内 /replica）。层文件按内容哈希命名并在所有同步之间共享：同一来源 epoch
再次同步另一分支时，已复制的祖先层直接复用、绝不重写。

磁盘布局（REPLICA_DIR）：
    manifest.json    副本清单：disk_size 与每条已提交的同步链（原子替换更新）
    journal.json     同步操作的两阶段日志，仅在操作进行中存在
    layers/<sha256>.layer   内容寻址的层文件，逐字节复制自源卷，写一次后不可变

一致性 —— 复制或写清单失败、进程重启后，备用卷只呈现“同步前”或“完整新链”：
    1. 逐字节复制缺失的层文件（已存在的哈希文件复用，tmp+rename+fsync）
    2. 写 journal.json（新文件清单 + 完整新清单）
    3. 原子替换 manifest.json                      —— 唯一提交点
    4. 删除 journal（收尾，失败留给启动恢复）

启动恢复：
    journal 有效且 manifest 已含本次 sync_id -> 前滚：删除 journal
    journal 有效且 manifest 不含 sync_id     -> 回滚：删除本次新文件，旧链原样保留
    journal 缺失/损坏                        -> 切换未开始，清理临时/孤儿文件
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .store import (
    BLOCK_SIZE,
    CorruptStoreError,
    NotFoundError,
    StoreError,
    _atomic_write,
    _fsync_dir,
)

REPLICA_MANIFEST = "manifest.json"
REPLICA_JOURNAL = "journal.json"
REPLICA_LAYERS_DIR = "layers"
MANIFEST_VERSION = 1


class ReplicaStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.layers_dir = self.root / REPLICA_LAYERS_DIR
        self.layers_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.recovery_events: list[str] = []
        self._recover()
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------ #
    # 启动恢复
    # ------------------------------------------------------------------ #
    def _recover(self) -> None:
        journal_path = self.root / REPLICA_JOURNAL
        journal = None
        if journal_path.exists():
            try:
                journal = json.loads(journal_path.read_text("utf-8"))
                if (not isinstance(journal, dict)
                        or journal.get("op") != "replica-sync"
                        or "sync_id" not in journal
                        or "new_manifest" not in journal):
                    raise ValueError("journal 结构不完整")
            except Exception:
                # 日志先于清单切换写入；日志损坏意味着崩溃发生在提交点之前，
                # 已提交的旧链必然完好，按“无在途操作”处理并清掉残迹。
                journal = None
                journal_path.unlink(missing_ok=True)
                self.recovery_events.append(
                    "副本卷发现损坏的 journal（崩溃发生在切换前），已按无在途操作清理"
                )

        if journal is not None:
            manifest = self._load_manifest()
            committed = manifest is not None and any(
                chain.get("sync_id") == journal["sync_id"]
                for chain in manifest.get("chains", [])
            )
            if committed:
                # 清单已切换、仅收尾未完成 -> 前滚（同步只增文件，无旧文件要删）
                journal_path.unlink(missing_ok=True)
                self.recovery_events.append(
                    f"副本卷检测到已切换未收尾的同步（目标 {journal.get('target')}），"
                    "已前滚并清理 journal"
                )
            else:
                # 清单尚未切换 -> 回滚：删除本次同步新写的文件，旧链原样保留
                referenced = self._referenced_files(manifest)
                for name in journal.get("new_files", []):
                    if name not in referenced:
                        self._delete_layer_file(name)
                journal_path.unlink(missing_ok=True)
                self.recovery_events.append(
                    f"副本卷检测到未完成的同步（目标 {journal.get('target')}），"
                    "已回滚，保留同步前的链"
                )

        for tmp in self.root.glob("*.tmp-*"):
            tmp.unlink(missing_ok=True)
        for tmp in self.layers_dir.glob("*.tmp-*"):
            tmp.unlink(missing_ok=True)
        self._gc_orphans()

    def _gc_orphans(self) -> None:
        referenced = self._referenced_files(self._load_manifest())
        removed = 0
        for f in self.layers_dir.iterdir():
            if f.is_file() and f.name not in referenced:
                f.unlink(missing_ok=True)
                removed += 1
        if removed:
            self.recovery_events.append(f"副本卷清理了 {removed} 个未被引用的层文件")

    def _referenced_files(self, manifest: dict | None) -> set[str]:
        if not manifest:
            return set()
        referenced: set[str] = set()
        for chain in manifest.get("chains", []):
            for info in chain.get("layers", {}).values():
                referenced.add(info["file"])
        return referenced

    def _load_manifest(self) -> dict | None:
        path = self.root / REPLICA_MANIFEST
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception as exc:
            raise CorruptStoreError(f"副本 manifest.json 损坏：{exc}") from exc
        if (not isinstance(data, dict) or data.get("version") != MANIFEST_VERSION
                or not isinstance(data.get("chains"), list)):
            raise CorruptStoreError("副本 manifest.json 结构不完整")
        return data

    # ------------------------------------------------------------------ #
    # 层文件（内容寻址，不可变）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _hash_name(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest() + ".layer"

    def _delete_layer_file(self, name: str) -> None:
        try:
            (self.layers_dir / name).unlink(missing_ok=True)
        except OSError:
            pass  # 留待下次启动时的孤儿清理

    def _place_layer(self, name: str, payload: bytes) -> None:
        """把层文件放到备用卷。同名文件已存在则复用（校验哈希后直接返回）。"""
        target = self.layers_dir / name
        if target.exists():
            # 内容寻址：文件名即 sha256，存在即等价；仍校验一次防止外部损坏
            if hashlib.sha256(target.read_bytes()).hexdigest() + ".layer" != name:
                raise CorruptStoreError(f"副本层文件内容与名称不符：{name}")
            return
        tmp = target.with_name(f"{target.name}.tmp-{uuid.uuid4().hex}")
        try:
            with open(tmp, "wb") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
        _fsync_dir(self.layers_dir)

    # ------------------------------------------------------------------ #
    # 同步
    # ------------------------------------------------------------------ #
    def sync(self, snapshot: dict) -> dict:
        """把源卷导出的一致链快照复制进备用卷。

        提交点（manifest 原子替换）之前任何失败都只清理本次新文件，
        备用卷已有的链与源卷均不受影响。
        """
        with self._lock:
            plan = self._prepare_sync(snapshot)
            if plan is None:  # 幂等命中，已直接返回状态
                return self._idempotent_state(snapshot)
            try:
                self._commit_sync(plan)
            except OSError as exc:
                self._rollback_sync(plan)
                raise StoreError(
                    f"写副本清单失败，已回滚，备用卷同步前的链未受影响：{exc}"
                ) from exc
            self._finalize_sync(plan)
            state = self.state()
            state["message"] = (
                f"同步完成：恢复点 {plan['target']} 及其 {len(plan['chain_order']) - 1} "
                f"个祖先层已复制到备用卷；新复制 {len(plan['new_files'])} 层，"
                f"复用 {plan['reused']} 层"
            )
            state["reused_layers"] = plan["reused"]
            state["copied_layers"] = len(plan["new_files"])
            return state

    def _idempotent_state(self, snapshot: dict) -> dict:
        existing = next(
            c for c in self._manifest["chains"]
            if c["source_epoch"] == snapshot["source_epoch"]
            and c["target"] == snapshot["chain"][-1]
        )
        state = self.state()
        state["message"] = (
            f"恢复点 {snapshot['chain'][-1]} 的链已存在于备用卷"
            f"（同步 {existing['sync_id'][:8]}），全部层复用，未重复复制"
        )
        state["reused_layers"] = len(snapshot["chain"])
        state["copied_layers"] = 0
        return state

    def _prepare_sync(self, snapshot: dict) -> dict | None:
        """第 1-2 步：复制缺失层文件 + 写 journal。旧链完全不动。

        返回 None 表示幂等命中（链已存在，无需复制）。
        """
        chain = snapshot["chain"]
        source_epoch = snapshot["source_epoch"]
        disk_size = snapshot["disk_size"]
        src_layers = snapshot["layers"]
        payloads = snapshot["payloads"]

        manifest = self._manifest or {
            "version": MANIFEST_VERSION,
            "disk_size": disk_size,
            "chains": [],
        }
        if manifest["disk_size"] != disk_size:
            raise StoreError(
                f"备用卷现有链的盘大小为 {manifest['disk_size']} 字节，"
                f"与本次同步的 {disk_size} 字节不一致，拒绝混合同步"
            )
        for existing in manifest["chains"]:
            if (existing["source_epoch"] == source_epoch
                    and existing["target"] == chain[-1]):
                return None

        on_disk = {p.name for p in self.layers_dir.iterdir() if p.is_file()}
        new_files: list[str] = []
        replica_layers: dict[str, dict] = {}
        reused = 0
        try:
            for lid in chain:  # 根 -> 目标，保持原父子关系
                payload = payloads[lid]
                name = self._hash_name(payload)
                if name in on_disk or name in new_files:
                    reused += 1
                else:
                    self._place_layer(name, payload)
                    new_files.append(name)
                meta = src_layers[lid]
                replica_layers[lid] = {
                    "parent": meta["parent"],
                    "file": name,
                    "blocks": meta["blocks"],
                }
        except OSError as exc:
            for name in new_files:
                self._delete_layer_file(name)
            raise StoreError(
                f"复制层文件失败，已回滚，备用卷同步前的链未受影响：{exc}"
            ) from exc

        sync_id = uuid.uuid4().hex
        entry = {
            "sync_id": sync_id,
            "source_epoch": source_epoch,
            "target": chain[-1],
            "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "disk_size": disk_size,
            "chain_order": list(chain),
            "layers": replica_layers,
        }
        new_manifest = {
            "version": MANIFEST_VERSION,
            "disk_size": disk_size,
            "chains": manifest["chains"] + [entry],
        }
        plan = {
            "op": "replica-sync",
            "sync_id": sync_id,
            "source_epoch": source_epoch,
            "target": chain[-1],
            "new_files": new_files,
            "reused": reused,
            "chain_order": list(chain),
            "new_manifest": new_manifest,
        }
        try:
            _atomic_write(
                self.root / REPLICA_JOURNAL,
                json.dumps(plan, ensure_ascii=False).encode("utf-8"),
            )
        except OSError as exc:
            for name in new_files:
                self._delete_layer_file(name)
            raise StoreError(
                f"写副本 journal 失败，已回滚，备用卷同步前的链未受影响：{exc}"
            ) from exc
        return plan

    def _commit_sync(self, plan: dict) -> None:
        """第 3 步：原子替换副本 manifest —— 唯一的提交点。"""
        _atomic_write(
            self.root / REPLICA_MANIFEST,
            json.dumps(plan["new_manifest"], ensure_ascii=False, indent=1).encode("utf-8"),
        )
        self._manifest = plan["new_manifest"]

    def _finalize_sync(self, plan: dict) -> None:
        """第 4 步：收尾删 journal。失败不影响一致性，留给启动恢复前滚。"""
        try:
            (self.root / REPLICA_JOURNAL).unlink(missing_ok=True)
        except OSError:
            pass

    def _rollback_sync(self, plan: dict) -> None:
        referenced = self._referenced_files(self._manifest)
        for name in plan["new_files"]:
            if name not in referenced:
                self._delete_layer_file(name)
        try:
            (self.root / REPLICA_JOURNAL).unlink(missing_ok=True)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # 查询 / 下载
    # ------------------------------------------------------------------ #
    def _find_chain(self, sync_id: str) -> dict:
        if self._manifest is None:
            raise NotFoundError("备用卷尚无同步副本")
        for chain in self._manifest["chains"]:
            if chain["sync_id"] == sync_id:
                return chain
        raise NotFoundError(f"副本链 {sync_id} 不存在")

    def resolve_image(self, sync_id: str, layer_id: str) -> bytes:
        """沿副本链的父子关系合并稀疏块，还原该恢复点的整盘原始字节。"""
        with self._lock:
            chain = self._find_chain(sync_id)
            layers = chain["layers"]
            if layer_id not in layers:
                raise NotFoundError(
                    f"恢复点 {layer_id} 不在副本链 {sync_id[:8]} 中"
                )
            order: list[str] = []
            cur: str | None = layer_id
            while cur is not None:
                if cur not in layers:
                    raise CorruptStoreError(
                        f"副本链 {sync_id[:8]} 缺少层 {cur}，链不完整"
                    )
                order.append(cur)
                cur = layers[cur]["parent"]
            merged: dict[int, bytes] = {}
            for name in reversed(order):  # 根 -> 目标，逐层覆盖
                path = self.layers_dir / layers[name]["file"]
                try:
                    raw = json.loads(path.read_text("utf-8"))
                except FileNotFoundError as exc:
                    raise CorruptStoreError(
                        f"副本层文件丢失：{layers[name]['file']}"
                    ) from exc
                except Exception as exc:
                    raise CorruptStoreError(
                        f"副本层文件损坏：{layers[name]['file']}: {exc}"
                    ) from exc
                for key, value in raw["blocks"].items():
                    merged[int(key)] = base64.b64decode(value)
            buf = bytearray(chain["disk_size"])
            for idx, data in merged.items():
                buf[idx * BLOCK_SIZE:(idx + 1) * BLOCK_SIZE] = data
            return bytes(buf)

    def state(self) -> dict:
        with self._lock:
            base = {
                "path": str(self.root),
                "recovery_events": list(self.recovery_events),
            }
            if self._manifest is None:
                return {**base, "initialized": False, "disk_size": None,
                        "layer_file_count": 0, "chains": []}
            files = self._referenced_files(self._manifest)
            chains = []
            for chain in self._manifest["chains"]:
                chains.append({
                    "sync_id": chain["sync_id"],
                    "source_epoch": chain["source_epoch"],
                    "target": chain["target"],
                    "synced_at": chain["synced_at"],
                    "disk_size": chain["disk_size"],
                    "layer_count": len(chain["layers"]),
                    "layers": [
                        {
                            "id": lid,
                            "parent": chain["layers"][lid]["parent"],
                            "blocks": chain["layers"][lid]["blocks"],
                            "file": chain["layers"][lid]["file"],
                        }
                        for lid in chain["chain_order"]
                    ],
                })
            return {
                **base,
                "initialized": True,
                "disk_size": self._manifest["disk_size"],
                "layer_file_count": len(files),
                "chains": chains,
            }
