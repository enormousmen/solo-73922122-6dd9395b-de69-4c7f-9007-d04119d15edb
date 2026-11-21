"""备用卷副本引擎测试：复制、层复用、稀疏保持、崩溃恢复、回滚与校验。"""

from __future__ import annotations

import json

import pytest

import app.replica as replica_mod
from app.replica import REPLICA_JOURNAL, ReplicaStore
from app.store import Store, StoreError, NotFoundError
from tests.conftest import (
    DISK_SIZE,
    blk,
    expected_image,
    make_manifest,
    sample_layers,
)


@pytest.fixture
def source(tmp_path):
    s = Store(tmp_path / "source")
    s.import_manifest(make_manifest())
    return s


@pytest.fixture
def replica(tmp_path):
    return ReplicaStore(tmp_path / "replica")


def sync(replica, source, layer_id):
    return replica.sync(source.snapshot_chain(layer_id))


def replica_files(r: ReplicaStore) -> set[str]:
    return {p.name for p in r.layers_dir.iterdir() if p.is_file()}


def referenced_files(r: ReplicaStore) -> set[str]:
    return r._referenced_files(r._manifest)


# --------------------------------------------------------------------- #
# 基本复制语义
# --------------------------------------------------------------------- #
class TestSyncBasic:
    def test_sync_copies_chain_and_preserves_parent_links(self, source, replica):
        st = sync(replica, source, "leaf")
        chain = next(c for c in st["chains"] if c["target"] == "leaf")
        assert [l["id"] for l in chain["layers"]] == ["base", "mid", "left", "leaf"]
        parents = {l["id"]: l["parent"] for l in chain["layers"]}
        assert parents == {"base": None, "mid": "base", "left": "mid", "leaf": "left"}
        assert chain["layer_count"] == 4

    def test_every_replica_point_resolves_to_source_bytes(self, source, replica):
        sync(replica, source, "leaf")
        sync_id = replica._manifest["chains"][0]["sync_id"]
        layers = sample_layers()
        for lid in ["base", "mid", "left", "leaf"]:
            assert replica.resolve_image(sync_id, lid) == expected_image(layers, lid)

    def test_sparse_blocks_preserved(self, source, replica):
        sync(replica, source, "left")
        info = replica._manifest["chains"][0]["layers"]["left"]
        raw = json.loads((replica.layers_dir / info["file"]).read_text())
        # left 在源卷只显式写块0；复制后仍只含块0，未物化父层块（稀疏保持）
        assert set(raw["blocks"]) == {"0"}

    def test_layer_files_are_byte_identical_to_source(self, source, replica):
        sync(replica, source, "leaf")
        for chain in replica._manifest["chains"]:
            for lid, info in chain["layers"].items():
                src_name = source._manifest["layers"][lid]["file"]
                assert (replica.layers_dir / info["file"]).read_bytes() == \
                    (source.layers_dir / src_name).read_bytes()

    def test_layer_files_content_addressed(self, source, replica):
        sync(replica, source, "leaf")
        for name in replica_files(replica):
            import hashlib
            payload = (replica.layers_dir / name).read_bytes()
            assert name == hashlib.sha256(payload).hexdigest() + ".layer"

    def test_sync_leaf_does_not_include_sibling_branch(self, source, replica):
        sync(replica, source, "right")
        ids = set(replica._manifest["chains"][0]["layers"])
        assert ids == {"base", "mid", "right"}  # left/leaf 不在链上，不复制

    def test_sync_before_any_import_rejected(self, tmp_path):
        empty_source = Store(tmp_path / "empty-source")
        r = ReplicaStore(tmp_path / "replica")
        with pytest.raises(StoreError, match="尚未导入"):
            r.sync(empty_source.snapshot_chain("base"))

    def test_sync_unknown_point_rejected(self, source, replica):
        with pytest.raises(NotFoundError):
            sync(replica, source, "ghost")

    def test_disk_size_mismatch_rejected(self, source, replica, tmp_path):
        sync(replica, source, "base")
        other = Store(tmp_path / "other")
        other.import_manifest(make_manifest(
            disk_size=DISK_SIZE * 2,
            layers=[{"id": "only", "parent": None,
                     "blocks": {"0": blk(0x55)}}]))
        with pytest.raises(StoreError, match="盘大小"):
            sync(replica, other, "only")


# --------------------------------------------------------------------- #
# 层复用：同一来源 epoch 再同步另一分支
# --------------------------------------------------------------------- #
class TestLayerReuse:
    def test_second_branch_reuses_copied_layers(self, source, replica):
        first = sync(replica, source, "leaf")
        assert first["copied_layers"] == 4 and first["reused_layers"] == 0
        files_after_first = replica_files(replica)

        second = sync(replica, source, "right")  # 共享 base、mid
        assert second["copied_layers"] == 1      # 仅 right 是新层
        assert second["reused_layers"] == 2      # base、mid 复用
        assert len(second["chains"]) == 2
        # 备用卷只增加 1 个去重层文件
        assert replica_files(replica) - files_after_first == {
            replica._manifest["chains"][1]["layers"]["right"]["file"]}
        assert replica_files(replica) == referenced_files(replica)

    def test_shared_layers_point_to_same_file(self, source, replica):
        sync(replica, source, "leaf")
        sync(replica, source, "right")
        chains = replica._manifest["chains"]
        assert chains[0]["layers"]["base"]["file"] == \
            chains[1]["layers"]["base"]["file"]
        assert chains[0]["layers"]["mid"]["file"] == \
            chains[1]["layers"]["mid"]["file"]

    def test_idempotent_resync_same_point(self, source, replica):
        first = sync(replica, source, "leaf")
        files = replica_files(replica)
        again = sync(replica, source, "leaf")
        assert again["copied_layers"] == 0 and again["reused_layers"] == 4
        assert len(again["chains"]) == 1          # 不新增链条目
        assert replica_files(replica) == files
        assert "已存在" in again["message"]

    def test_reused_layer_still_downloads_correctly(self, source, replica):
        sync(replica, source, "leaf")
        sync(replica, source, "right")
        second_id = replica._manifest["chains"][1]["sync_id"]
        layers = sample_layers()
        # 第二条链上复用的 base/mid 仍能独立还原
        assert replica.resolve_image(second_id, "base") == expected_image(layers, "base")
        assert replica.resolve_image(second_id, "mid") == expected_image(layers, "mid")
        assert replica.resolve_image(second_id, "right") == expected_image(layers, "right")

    def test_new_epoch_same_ids_do_not_collide(self, source, replica, tmp_path):
        # 同一套层 id 但块内容不同的“另一来源 epoch”：内容哈希不同，文件互不覆盖
        sync(replica, source, "base")
        changed = Store(tmp_path / "changed")
        changed.import_manifest(make_manifest(
            layers=[{"id": "base", "parent": None,
                     "blocks": {"0": blk(0x77), "9": blk(0x88)}}]))
        st = sync(replica, changed, "base")
        assert st["copied_layers"] == 1
        assert replica_files(replica) == referenced_files(replica)
        new_id = replica._manifest["chains"][1]["sync_id"]
        img = replica.resolve_image(new_id, "base")
        assert img[:4096] == bytes([0x77]) * 4096
        assert img[9 * 4096:10 * 4096] == bytes([0x88]) * 4096


# --------------------------------------------------------------------- #
# 崩溃恢复（重启 = 新建 ReplicaStore 实例）
# --------------------------------------------------------------------- #
class TestReplicaCrashRecovery:
    def test_crash_before_commit_rolls_back(self, source, tmp_path):
        rdir = tmp_path / "replica"
        r1 = ReplicaStore(rdir)
        plan = r1._prepare_sync(source.snapshot_chain("leaf"))
        assert (rdir / REPLICA_JOURNAL).exists()
        new_files = set(plan["new_files"])
        assert new_files <= replica_files(r1)

        r2 = ReplicaStore(rdir)  # 重启
        assert any("回滚" in e for e in r2.recovery_events)
        assert not (rdir / REPLICA_JOURNAL).exists()
        assert r2.state()["initialized"] is False
        assert replica_files(r2).isdisjoint(new_files)  # 本次新文件已清理
        assert replica_files(r2) == set()

    def test_crash_after_commit_rolls_forward(self, source, replica, tmp_path):
        rdir = replica.root
        plan = replica._prepare_sync(source.snapshot_chain("leaf"))
        replica._commit_sync(plan)
        assert (rdir / REPLICA_JOURNAL).exists()

        r2 = ReplicaStore(rdir)
        assert any("前滚" in e for e in r2.recovery_events)
        assert not (rdir / REPLICA_JOURNAL).exists()
        st = r2.state()
        assert len(st["chains"]) == 1 and st["chains"][0]["target"] == "leaf"
        assert replica_files(r2) == referenced_files(r2)
        layers = sample_layers()
        sync_id = r2._manifest["chains"][0]["sync_id"]
        assert r2.resolve_image(sync_id, "leaf") == expected_image(layers, "leaf")

    def test_crash_after_first_chain_second_sync_rolls_back_keeping_first(
            self, source, replica):
        sync(replica, source, "right")
        first_files = replica_files(replica)
        plan = replica._prepare_sync(source.snapshot_chain("leaf"))

        r2 = ReplicaStore(replica.root)
        assert any("回滚" in e for e in r2.recovery_events)
        # 第一条链完整保留，第二次同步的新文件被清理
        assert len(r2.state()["chains"]) == 1
        assert replica_files(r2) == first_files
        layers = sample_layers()
        sync_id = r2._manifest["chains"][0]["sync_id"]
        assert r2.resolve_image(sync_id, "right") == expected_image(layers, "right")

    def test_corrupt_journal_treated_as_pre_switch(self, source, replica):
        replica._prepare_sync(source.snapshot_chain("leaf"))
        (replica.root / REPLICA_JOURNAL).write_text("{broken", encoding="utf-8")
        r2 = ReplicaStore(replica.root)
        assert r2.state()["initialized"] is False
        assert replica_files(r2) == set()

    def test_orphan_files_collected_on_boot(self, source, replica):
        sync(replica, source, "base")
        (replica.layers_dir / "stray.layer").write_text("{}")
        (replica.layers_dir / "x.tmp-1").write_text("y")
        r2 = ReplicaStore(replica.root)
        assert "stray.layer" not in replica_files(r2)
        assert replica_files(r2) == referenced_files(r2)


# --------------------------------------------------------------------- #
# 写盘失败：提交点前失败，同步前的链必须完好
# --------------------------------------------------------------------- #
class TestReplicaWriteFailure:
    def test_manifest_write_failure_keeps_previous_chain(self, source, replica,
                                                         monkeypatch):
        sync(replica, source, "right")
        before_files = replica_files(replica)
        real_write = replica_mod._atomic_write

        def fail_on_manifest(path, data):
            if path.name == replica_mod.REPLICA_MANIFEST:
                raise OSError("模拟写盘失败")
            return real_write(path, data)

        monkeypatch.setattr(replica_mod, "_atomic_write", fail_on_manifest)
        with pytest.raises(StoreError, match="未受影响"):
            sync(replica, source, "leaf")
        monkeypatch.undo()

        assert not (replica.root / REPLICA_JOURNAL).exists()
        assert replica_files(replica) == before_files
        assert len(replica.state()["chains"]) == 1

    def test_layer_copy_failure_aborts_before_journal(self, source, replica,
                                                      monkeypatch):
        real_place = ReplicaStore._place_layer

        def fail_place(self, name, payload):
            raise OSError("模拟备用卷写满")

        monkeypatch.setattr(ReplicaStore, "_place_layer", fail_place)
        with pytest.raises(StoreError, match="未受影响"):
            sync(replica, source, "leaf")
        monkeypatch.undo()

        assert not (replica.root / REPLICA_JOURNAL).exists()
        assert replica.state()["initialized"] is False
        assert replica_files(replica) == set()
        assert real_place  # 原方法仍在，重启后状态也一致
        assert ReplicaStore(replica.root).state()["initialized"] is False


# --------------------------------------------------------------------- #
# 源卷不受影响
# --------------------------------------------------------------------- #
class TestSourceUntouched:
    def test_sync_does_not_modify_source(self, source, replica):
        import hashlib
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in source.layers_dir.iterdir() if p.is_file()}
        sync(replica, source, "leaf")
        sync(replica, source, "right")
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in source.layers_dir.iterdir() if p.is_file()}
        assert before == after
        layers = sample_layers()
        for lid in ["base", "mid", "left", "right", "leaf"]:
            assert source.resolve_image(lid) == expected_image(layers, lid)

    def test_replica_download_unknown_chain_or_layer(self, source, replica):
        sync(replica, source, "right")
        with pytest.raises(NotFoundError):
            replica.resolve_image("deadbeef", "right")
        sync_id = replica._manifest["chains"][0]["sync_id"]
        with pytest.raises(NotFoundError):
            replica.resolve_image(sync_id, "leaf")  # 不在该链上
