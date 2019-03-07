class Constants:
    @staticmethod
    def get_emotion_cols():
        return ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']

    @staticmethod
    def get_label_cols(self):
        return ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

    @staticmethod
    def get_no_emotions():
        return len(Constants.get_emotion_cols())