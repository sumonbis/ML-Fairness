#!/bin/sh
if [ $# -eq 2 ]
then
  echo "Started running model $2 for the task $1"
  python $1/model$2.py
fi
if [ $# -eq 1 ]
then
  if [ $1 == 'all' ]
  then
     echo "Started running all the models for all the tasks."
     python titanic/model1.py
     python titanic/model2.py
     python titanic/model3.py
     python titanic/model4.py
     python titanic/model5.py
     python titanic/model6.py
     python titanic/model7.py
     python titanic/model8.py

     python adult/model1.py
     python adult/model2.py
     python adult/model3.py
     python adult/model4.py
     python adult/model5.py
     python adult/model6.py
     python adult/model7.py
     python adult/model8.py

     python german/model1.py
     python german/model2.py
     python german/model3.py
     python german/model4.py
     python german/model5.py
     python german/model6.py
     python german/model7.py
     python german/model8.py

     python bank/model1.py
     python bank/model2.py
     python bank/model3.py
     python bank/model4.py
     python bank/model5.py
     python bank/model6.py
     python bank/model7.py
     python bank/model8.py

     python home/model1.py
     python home/model2.py
     python home/model3.py
     python home/model4.py
     python home/model5.py
     python home/model6.py
     python home/model7.py
     python home/model8.py
  else
    echo "Started running all the models for" $1
    python $1/model1.py
    python $1/model2.py
    python $1/model3.py
    python $1/model4.py
    python $1/model5.py
    python $1/model6.py
    python $1/model7.py
    python $1/model8.py
  fi
fi
