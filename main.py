from LoadData import load_data as ld
from LoadData import match
from EDA import eda
from ai import nn

if __name__ == '__main__':
    # load and pre-process the data #
    # load the truth for the test data
    test = ld(r'data/test.csv', index='PassengerId',
        drop=['Ticket', 'Cabin', 'Name'],
        columns={
            'Pclass': 'TicketClass',
            'SibSp': 'NumbSiblingsOrSpouse',
            'Parch': 'NumbParentsOrChildren',
        },
        norm=['Fare',
                'Age',
                'NumbSiblingsOrSpouse',
                'NumbParentsOrChildren'
            ],
        oneHot=['Embarked',
                'Sex',
                'TicketClass']
    )
    ytest = ld(r'data/gender_submission.csv', index='PassengerId')

    # remove entries filtered out
    ytest = match(test, ytest)

    # load the train data
    train = ld(
        r'data/train.csv',
        # index='Survived',
        drop=['PassengerId', 'Ticket', 'Cabin', 'Name'],
        columns={
            'Pclass': 'TicketClass',
            'SibSp': 'NumbSiblingsOrSpouse',
            'Parch': 'NumbParentsOrChildren',
        },
        norm=['Fare',
                'Age',
                'NumbSiblingsOrSpouse',
                'NumbParentsOrChildren'
            ],
        oneHot=['Embarked',
                'Sex',
                'TicketClass']
    )

    # eda(data=None)
    eda(data=train)

    nn(train, test, ytest)
