import pandas as pd
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

most_common_port = train_data['Embarked'].mode()[0]
train_data['Embarked'] = train_data['Embarked'].fillna(most_common_port)

median_age = train_data['Age' ].median()
train_data['Age' ] = train_data['Age' ].fillna(median_age)

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

train_data = pd.get_dummies(train_data, columns=['Sex' , 'Embarked'])

X = train_data.drop('Survived', axis = 1)
y = train_data['Survived']

test_passenger_ids = test_data['PassengerId']

most_common_port = test_data['Embarked'].mode()[0]
test_data['Embarked'] = test_data['Embarked'].fillna(most_common_port)

median_age = test_data['Age' ].median()
test_data['Age' ] = test_data['Age' ].fillna(median_age)

median_fare = train_data['Fare' ].median()
test_data['Fare' ] = test_data['Fare' ].fillna(median_age)

test_data = test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

test_data = pd.get_dummies(test_data, columns=['Sex' , 'Embarked'])

final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)
test_predictions = final_model.predict(test_data)

submission_data = pd.DataFrame({
    'PassengerId' : test_passenger_ids,
    'Survived': test_predictions
})

submission_data.to_csv('submission.csv', index = False)

print(" 'submission.csv file has been created successfully! ")







