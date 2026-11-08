"""Sentence splitting and criteria parsing. Run: python -m tests.test_app_text"""
from src.app.text import number_note, parse_criteria, split_sentences

assert split_sentences("Dr. Lee saw her. She took 3.5 mg daily. No fever!") == ["Dr. Lee saw her.", "She took 3.5 mg daily.", "No fever!"]
assert split_sentences("Takes drugs, e.g. Aspirin, daily. BMI is 30.") == ["Takes drugs, e.g. Aspirin, daily.", "BMI is 30."]
assert split_sentences("He took 5 mg. She did not.") == ["He took 5 mg.", "She did not."]
assert split_sentences("0. First one.\n1. Second one.") == ["First one.", "Second one."]
assert split_sentences("   \n\n ") == []
assert number_note(["A.", "B."]) == "0. A.\n1. B."

parsed = parse_criteria("Inclusion Criteria:\n\n* Age over 18\n1. Male\n\nExclusion Criteria:\n\n- Pregnancy\n* None")
assert parsed == [{"type": "inclusion", "text": "Age over 18"}, {"type": "inclusion", "text": "Male"}, {"type": "exclusion", "text": "Pregnancy"}]
assert parse_criteria("Inclusion Criteria:\n* Age \\>= 18")[0]["text"] == "Age >= 18"
try:
    parse_criteria("just some text")
    raise AssertionError("should have rejected text with no headers")
except ValueError:
    pass
print("all app text tests passed")
