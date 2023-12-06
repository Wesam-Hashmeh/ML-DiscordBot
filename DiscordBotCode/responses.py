import summarizer
import transFrench
import transEng


def handle_response(message, user) -> str:
    if message == 'Hello':
        return "Hey"

    if message[:3] == '!st':
        return summarizer.summarize_text(message[4:])

    if message[:4] == '!e2f':
        return f'**{user} says** : {transFrench.translate_english_to_french(message[5:])}'

    if message[:4] == '!f2e':
        return f'**{user} says** : {transEng.translate_french_to_english(message[5:])}'

    if message == '!help':
        return ("Here are the commands that you can use\n"
                "`!f2e [Message] => Translates French to English Text\n"
                "!e2f [Message] => Translates English to French Text\n"
                "!st [Message] => Summarizes a large input of Text`")

    return 'I did not understand the prompt, try again or use !help.'
