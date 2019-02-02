function [Accurecy, Recall, Precision, FScore] = TestPerformance(y, y_predicted)

length0= length(y);

for i=1 : length0
   yNew(1,i)=y(i,1);
   y_predictedNew(1,i)=y_predicted(i,1);    
end

[c,cm,ind,per] = confusion(yNew,y_predictedNew);

TN=cm(1,1);
FP=cm(1,2);
FN=cm(2,1);
TP=cm(2,2);

ALL=(FP+FN+TP+TN);

Accurecy=((TP+TN)/ALL)*100
Recall= (TP/(TP+FN))*100
Precision= (TP/(TP+FP))*100
FScore= (2*(Precision*Recall)/(Precision+Recall))*100

end
