class Constants:
    @staticmethod
    def get_emotion_cols():
        return ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown']

    @staticmethod
    def get_label_cols():
        return ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt', 'unknown', 'NF']

    @staticmethod
    def get_no_emotions():
        return len(Constants.get_emotion_cols())

    @staticmethod
    def correct_emotion_label(inp):
        # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
        emotion_dict = {0: 4, 1: 5, 2: 6, 3: 1, 4: 3, 5: 2, 6: 0}
        return emotion_dict[inp]

    @staticmethod
    def get_output_image_size():
        return 160

    @staticmethod
    def get_max_FEC_annotations():
        """
        This number is selected based on the train and test set.
        Needed for proper reading of the dataset as a pandas dataframe.
        """
        return 11