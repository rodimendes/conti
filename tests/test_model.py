from main_folder import ask_model

def test_generate_text():
    """
    Dummy example of code to check if it returns a string
    """
    model = ["databricks/dolly-v2-7b"]
    result = ask_model("What is Python?", model)

    assert type(result[0]["generated_text"]) == "str"