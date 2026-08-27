import uuid

from app.retrieval.qdrant_store import chunk_point_id


def test_same_chunk_id_always_maps_to_same_point_id():
    # Re-ingestion must upsert (overwrite), never duplicate - see
    # app/retrieval/qdrant_store.py's module docstring.
    a = chunk_point_id("pci_dss_v4.0.1::8.4.2::requirement")
    b = chunk_point_id("pci_dss_v4.0.1::8.4.2::requirement")
    assert a == b
    assert uuid.UUID(a)  # valid UUID string


def test_different_chunk_ids_map_to_different_point_ids():
    a = chunk_point_id("pci_dss_v4.0.1::8.4.2::requirement")
    b = chunk_point_id("pci_dss_v4.0.1::8.4.3::requirement")
    assert a != b
