import pandas as pd
from matplotlib import pyplot as plt


def eda(data: pd.DataFrame = None):
    return

    if data is None:
        return
    
    lived = data[data['Survived'] == 1]
    died = data[data['Survived'] == 0]

    # building custom plots

    # box plots for numerical data
    for col in ['Age', 'NumbSiblingsOrSpouse', 'NumbParentsOrChildren', 'Fare']:
        plt.boxplot([lived[col], died[col]], labels=['lived', 'died'])
        plt.xlabel('lived 1, died 0')
        plt.ylabel(col)
        plt.show()

    # box plots counting how many s
    plt.bar(1, lived['Survived'].value_counts())
    plt.bar(0, died['Survived'].value_counts())
    plt.xlabel('lived 1, died 0')
    plt.ylabel('Survived')
    plt.title('freq bar graph for survivors')
    plt.show()

    # box plots counting frequency for categorical data
    for col in [['Embarked_C', 'Embarked_Q', 'Embarked_S'],
                ['Sex_female', 'Sex_male'],
                ['TicketClass_1', 'TicketClass_2', 'TicketClass_3']]:
        w = 0.2
        offset = - w
        for sub_col in col:
            plt.bar(1 - offset, lived[sub_col].value_counts(), width=w, label=sub_col.split("_")[1])
            plt.bar(0 - offset, died[sub_col].value_counts(), width=w, label=sub_col.split("_")[1])
            offset += w
        plt.xlabel('lived 1, died 0')
        plt.ylabel(col[0].split("_")[0])
        plt.legend()
        plt.show()
