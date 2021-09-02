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
    "import numpy as np\n",
    "from sklearn import datasets\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "#Load dataset\n",
    "wine = datasets.load_wine()"
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
      "Features:  ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']\n",
      "Labels:  ['class_0' 'class_1' 'class_2']\n",
      "\n",
      "Data:  [[1.423e+01 1.710e+00 2.430e+00 ... 1.040e+00 3.920e+00 1.065e+03]\n",
      " [1.320e+01 1.780e+00 2.140e+00 ... 1.050e+00 3.400e+00 1.050e+03]\n",
      " [1.316e+01 2.360e+00 2.670e+00 ... 1.030e+00 3.170e+00 1.185e+03]\n",
      " ...\n",
      " [1.327e+01 4.280e+00 2.260e+00 ... 5.900e-01 1.560e+00 8.350e+02]\n",
      " [1.317e+01 2.590e+00 2.370e+00 ... 6.000e-01 1.620e+00 8.400e+02]\n",
      " [1.413e+01 4.100e+00 2.740e+00 ... 6.100e-01 1.600e+00 5.600e+02]]\n",
      "\n",
      "Target:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n",
      " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]\n"
     ]
    }
   ],
   "source": [
    "# print the names of the features\n",
    "print(\"Features: \", wine.feature_names)\n",
    "\n",
    "# print the label type of wine(class_0, class_1, class_2)\n",
    "print(\"Labels: \", wine.target_names)\n",
    "\n",
    "# print data(feature)shape\n",
    "wine.data.shape\n",
    "\n",
    "print(\"\\nData: \",wine.data)\n",
    "print(\"\\nTarget: \",wine.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "#split data set into train and test sets\n",
    "data_train, data_test, target_train, target_test = train_test_split(wine.data,\n",
    "                        wine.target, test_size = 0.34, random_state = 105)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "gnb = GaussianNB()\n",
    "\n",
    "#Train the model using the training sets\n",
    "gnb.fit(data_train, target_train)\n",
    "\n",
    "#Predict the response for test dataset\n",
    "target_pred = gnb.predict(data_test)"
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
      "Accuracy: 0.9508196721311475\n"
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
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[22,  0,  0],\n",
       "       [ 2, 21,  1],\n",
       "       [ 0,  0, 15]], dtype=int64)"
      ]
     },
     "execution_count": 6,
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
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "precision: [0.91666667 1.         0.9375    ]\n",
      "recall: [1.    0.875 1.   ]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "\n",
    "precision = precision_score(target_test, target_pred, average=None)\n",
    "recall = recall_score(target_test, target_pred, average=None)\n",
    "\n",
    "print('precision: {}'.format(precision))\n",
    "print('recall: {}'.format(recall))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
