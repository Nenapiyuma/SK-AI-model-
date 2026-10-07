from model.parameter_audit import audit_1_3t, storage_bytes, int4_block_storage_bytes

def test_exact_parameter_count():
    assert audit_1_3t().total_parameters == 1300000000000

def test_active_parameters_are_lower_than_total():
    a = audit_1_3t()
    assert 0 < a.active_parameters < a.total_parameters

def test_storage_accounting():
    p = 1300000000000
    assert storage_bytes(p, 16) == 2600000000000
    assert storage_bytes(p, 8) == 1300000000000
    assert storage_bytes(p, 4) == 650000000000
    assert int4_block_storage_bytes(p, 128, 2) > storage_bytes(p, 4)
