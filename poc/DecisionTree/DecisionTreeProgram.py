import pandas as pd
import sys
from six import StringIO
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import pydotplus


def ai_image(ai, data, target):
    dot_data = StringIO()
    export_graphviz(ai, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=['axis1', 'axis2', 'axis3'], class_names=['Asleep', 'Awake'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png(sys.argv[1] + ".png")


def ai_guess(ai, x_test, y_test):
    print("starting AI testing")
    y_pred = ai.predict(x_test)
    print("Ai tested.")
    print(x_test)
    print(y_pred)
    accuracy_result = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy of the AI is:", accuracy_result)
    return accuracy_result


def ai_train(data, target):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

    print("starting AI training")
    clf = DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(x_train, y_train)
    print("Ai trained. testing it")
    ai_guess(clf, x_test, y_test)
    ai_image(clf, data, target)



def separate_csv(csv):
    csv.drop(['Number', 'timestamp', 'count'], axis=1, inplace=True)  # dropping useless part of the csv
    csv.head()
    data_row = csv[['axis1', 'axis2', 'axis3']]
    target_row = csv[['sleep']]
    print(data_row)
    print(target_row)
    ai_train(data_row, target_row)


def real_main(argv):
    if len(argv) < 2:
        print("Usage: ./program [path to csv]")
        exit(-1)
    dataset = pd.read_csv(argv[1])
    separate_csv(dataset)


if __name__ == '__main__':
    real_main(sys.argv)
