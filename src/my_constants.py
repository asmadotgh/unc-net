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

    @staticmethod
    def get_embedding_tensor_name(inp):
        embedding_dict = {'Mixed_8b': 'InceptionResnetV1/Logits/AvgPool_1a_8x8/AvgPool:0',
                          'Mixed_8a': 'InceptionResnetV1/Block8:0', 'Mixed_7a': 'InceptionResnetV1/Repeat2:0',
                          'Mixed_6b': 'InceptionResnetV1/Mixed_7a:0', 'Mixed_6a': 'InceptionResnetV1/Repeat1:0',
                          'Mixed_5a': 'InceptionResnetV1/Mixed_6a:0', 'default': 'embeddings:0'}
        for key in embedding_dict.keys():
            if key == inp:
                return embedding_dict[key]
        print(f'Embedding {inp} not supported. Returning default embeddings:0.')
        return 'embeddings:0'
