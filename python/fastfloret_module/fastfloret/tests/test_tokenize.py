import fastfloret as floret


def test_tokenize():
    assert floret.tokenize("a b cde") == ["a", "b", "cde"]
