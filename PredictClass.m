function y = PredictClass(x, theta, threshold)

H_theta= sigmoid(x*theta);

length1 = length(H_theta);

for i=1 : length1

    if (H_theta(i,1) > threshold || H_theta(i,1) == threshold)
     y(i,1) = 1;
    else 
     y(i,1) = 0;
    end

end

end
