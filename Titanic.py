import pandas as pd
from model import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
from time import time

data_raw = pd.read_csv("data/train.csv", index_col='PassengerId')
data_validate = pd.read_csv("data/test.csv", index_col='PassengerId')
data_raw.sample(10)
data_raw.info()
data_raw.isnull().sum()
data_raw.describe(include='all')
data_raw['Sex'].value_counts()
data_raw['Embarked'].value_counts()
data_copy = data_raw.copy(deep=True)
data_cleaner = [data_copy, data_validate]

for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace = True)
    dataset.drop(['Cabin', 'Ticket', 'Fare', 'Name'], axis=1, inplace = True)

for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    # We set IsAlone to 1/True for everyone and then change it to 0/False depending on their FamilySize.
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0
    dataset.drop(['SibSp', 'Parch'], axis=1, inplace = True)


data_cleaner[0].head()


for dataset in data_cleaner:
    dataset['Sex'].loc[dataset['Sex'] == 'male'] = 0
    dataset['Sex'].loc[dataset['Sex'] == 'female'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'C'] = 0
    dataset['Embarked'].loc[dataset['Embarked'] == 'Q'] = 1
    dataset['Embarked'].loc[dataset['Embarked'] == 'S'] = 2



data_cleaner[0].head()


data_clean, data_validate = data_cleaner
data_labels = data_clean['Survived']
data_features = data_clean.drop('Survived', axis=1)


features_train, features_test, labels_train, labels_test = train_test_split(data_features, data_labels,
                                                                            test_size=0.2, random_state=42)

features_train.head()
labels_train.head()
features_test.head()
labels_test.head()
data_validate.head()
nb_classifier = GaussianNB()
t0 = time()
nb_classifier.fit(features_train, labels_train)
print("Training Time: ", time()-t0, "s.", sep='')

t1 = time()
nb_pred = nb_classifier.predict(features_test)
print("Testing Time: ", time()-t1, "s.", sep='')


print("Accuracy: ", accuracy_score(labels_test, nb_pred), ".", sep='')


dt_classifier = tree.DecisionTreeClassifier(min_samples_split=40)

t0 = time()
dt_classifier.fit(features_train, labels_train)
print("Training Time: ", round(time() - t0), "s")

t1 = time()
dt_prediction = dt_classifier.predict(features_test)
print("Prediction Time: ", round(time() - t1), "s")

print(accuracy_score(labels_test, dt_prediction))

features_test.head()

dt_classifier.predict(features_test.head())

labels_test[:5]

final = dt_classifier.predict(data_validate)

sample = pd.read_csv("data/sample.csv", index_col='PassengerId')
sample['Survived'] = final
sample.to_csv("data/submission.csv", )
