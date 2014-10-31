import data_io

def main():
    print("Reading test data")
    test = data_io.read_test()
    test.fillna(0, inplace=True)

    feature_names = list(test.columns)
    feature_names.remove("Tweet")
    feature_names.remove("JJ_count")
    feature_names.remove("NN_count")
    feature_names.remove("VB_count")
    feature_names.remove("RB_count")
    print len(feature_names)
    features = test[feature_names].values

    print("Loading the classifier")
    classifier = data_io.load_model()

    print("Making predictions")
    print features.shape
    predictions = classifier.predict(features)
    #predictions = list(-1.0*predictions)
    print predictions
    recommendations = zip(test["Tweet"], predictions)

    print("Writing predictions to file")
    data_io.write_submission(recommendations)

if __name__=="__main__":
    main()
