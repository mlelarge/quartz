"""Incremental detokenizer: multibyte hold-back + stop sequences."""

from quartz.detokenize import IncrementalDetokenizer


class FakeByteTokenizer:
    """Minimal tokenizer: each id maps to a byte; decode() does UTF-8 with the
    'replace' error handler, so an incomplete multibyte char yields U+FFFD."""

    def __init__(self, id_to_byte):
        self._m = id_to_byte

    def decode(self, ids):
        return bytes(self._m[i] for i in ids).decode("utf-8", errors="replace")


def test_multibyte_held_back_until_complete():
    # '€' is 3 UTF-8 bytes: E2 82 AC
    tok = FakeByteTokenizer({0: 0xE2, 1: 0x82, 2: 0xAC})
    d = IncrementalDetokenizer(tok)
    assert d.add_token(0) == ""        # incomplete -> withheld
    assert d.add_token(1) == ""        # still incomplete
    out = d.add_token(2) + d.finalize()
    assert "€" in out
    assert "�" not in out


def test_stop_sequence_cuts_and_holds():
    # ascii: 'E'=69 'N'=78 'D'=68 'x'=120
    tok = FakeByteTokenizer({0: 69, 1: 78, 2: 68, 3: 120})
    d = IncrementalDetokenizer(tok, stop=("END",))
    # hold-back = len("END")-1 = 2, so 'E' and 'EN' are withheld
    d.add_token(3)                     # 'x' -> emitted (beyond hold window once more comes)
    d.add_token(0)                     # 'E'
    d.add_token(1)                     # 'EN'
    seg = d.add_token(2)               # 'END' completes the stop
    assert d.finished and d.stop_reason == "stop"
    assert "END" not in (seg or "")    # stop string itself is not emitted
