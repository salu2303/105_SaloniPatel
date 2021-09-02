{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Experiment 3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "import pandas as pd\n",
    "import numpy as np "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   Outlook Temp Wind  Humidity  Class\n",
      "0        R    H    F         1      0\n",
      "1        R    H    T         2      0\n",
      "2        O    H    F         1      1\n",
      "3        R    M    F         1      1\n",
      "4        S    C    F         1      1\n",
      "5        O    C    T         0      0\n",
      "6        O    C    T         1      1\n",
      "7        R    M    F         1      0\n",
      "8        O    C    F         0      1\n",
      "9        S    M    F         2      1\n",
      "10       R    C    T         2      0\n",
      "11       O    M    T         0      1\n",
      "12       O    H    F         1      1\n",
      "13       S    M    T         1      0\n"
     ]
    }
   ],
   "source": [
    "datasets = pd.read_csv('Dataset3.csv') \n",
    "X=datasets.iloc[:,:-1].values\n",
    "print(datasets)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "OneHotEncoder(categorical_features=None, categories=None, drop=None,\n",
       "              dtype=<class 'numpy.float64'>, handle_unknown='ignore',\n",
       "              n_values=None, sparse=True)"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import OneHotEncoder\n",
    "enc = OneHotEncoder(handle_unknown='ignore')\n",
    "enc.fit(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[array(['O', 'R', 'S'], dtype=object),\n",
       " array(['C', 'H', 'M'], dtype=object),\n",
       " array(['F', 'T'], dtype=object),\n",
       " array([0, 1, 2], dtype=object)]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "enc.categories_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "features: \n",
      " [[0 1 0 0 1 0 1 0 0 1 0]\n",
      " [0 1 0 0 1 0 0 1 0 0 1]\n",
      " [1 0 0 0 1 0 1 0 0 1 0]\n",
      " [0 1 0 0 0 1 1 0 0 1 0]\n",
      " [0 0 1 1 0 0 1 0 0 1 0]\n",
      " [1 0 0 1 0 0 0 1 1 0 0]\n",
      " [1 0 0 1 0 0 0 1 0 1 0]\n",
      " [0 1 0 0 0 1 1 0 0 1 0]\n",
      " [1 0 0 1 0 0 1 0 1 0 0]\n",
      " [0 0 1 0 0 1 1 0 0 0 1]\n",
      " [0 1 0 1 0 0 0 1 0 0 1]\n",
      " [1 0 0 0 0 1 0 1 1 0 0]\n",
      " [1 0 0 0 1 0 1 0 0 1 0]\n",
      " [0 0 1 0 0 1 0 1 0 1 0]]\n",
      "class: \n",
      " [0 0 1 1 1 0 1 0 1 1 0 1 1 0]\n"
     ]
    }
   ],
   "source": [
    "data=enc.transform(X).toarray()\n",
    "features = data.astype(int)\n",
    "class_col=np.array(datasets[\"Class\"])\n",
    "print(\"features: \\n\" ,features)\n",
    "print(\"class: \\n\" ,class_col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "#split data set into train and test sets\n",
    "data_train, data_test, target_train, target_test = train_test_split(features,\n",
    "                        class_col, test_size = 0.15, random_state = 105)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.naive_bayes import GaussianNB, MultinomialNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=MultinomialNB()\n",
    "\n",
    "#Train the model using the training sets\n",
    "model.fit(data_train, target_train)\n",
    "\n",
    "#Predict the response for test dataset\n",
    "target_pred = model.predict(data_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy: 0.6666666666666666\n"
     ]
    }
   ],
   "source": [
    "#Import scikit-learn metrics module for accuracy calculation\n",
    "from sklearn import metrics\n",
    "\n",
    "# Model Accuracy, how often is the classifier correct?\n",
    "print(\"Accuracy:\",metrics.accuracy_score(target_test, target_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1, 0],\n",
       "       [1, 1]], dtype=int64)"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Import confusion_matrix from scikit-learn metrics module for confusion_matrix\n",
    "from sklearn.metrics import confusion_matrix\n",
    "confusion_matrix(target_test, target_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "precision: 1.0\n",
      "recall: 0.5\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "\n",
    "precision = precision_score(target_test, target_pred)\n",
    "recall = recall_score(target_test, target_pred)\n",
    "\n",
    "print('precision: {}'.format(precision))\n",
    "print('recall: {}'.format(recall))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Value: [0]\n"
     ]
    }
   ],
   "source": [
    "#if Outlook is ’0 1 0 = Rainy’, Temperature is ’0 0 1 = Mild’, Humidity =’0 1 0=Normal’, and Wind = ’0 1=False’\n",
    "predicted= model.predict([[0,1, 0, 0, 0, 1, 0, 1, 0, 1, 0]])\n",
    "print(\"Predicted Value:\", predicted)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Predicted Value: [1]\n"
     ]
    }
   ],
   "source": [
    "#Outlook is ’001=Sunny’, Temeprature is ’100=Cool’, Humidity =’0 0 1High’, and Wind = ’10 true’?\n",
    "predicted= model.predict([ [0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1]]) \n",
    "print(\"Predicted Value:\", predicted)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Krinish Radadiya ( CE107 )"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
