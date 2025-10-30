
class Clarity:
    CLEAR_REPLY = "Clear Reply"
    AMBIVALENT = "Ambivalent"
    CLEAR_NON_REPLY ="Clear Non-Reply"

class Interview:
    def __init__(self, interviews):
        self.interviews = interviews

    def getQuestionAnswer(self):
        return [f"{row['interview_question']} {row['interview_answer']}" for row in self.interviews]

    def getClarity(self):
        return [row["clarity_label"] for row in self.interviews]