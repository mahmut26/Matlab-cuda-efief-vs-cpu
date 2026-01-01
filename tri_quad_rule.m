function [qp, qw] = tri_quad_rule(order)
% Ref tri: (0,0),(1,0),(0,1), area=1/2
switch order
    case 1
        qp = [1/3, 1/3];
        qw = 1/2;
    case 2
        qp = [1/6, 1/6;
              2/3, 1/6;
              1/6, 2/3];
        qw = (1/6) * ones(3,1); % sum=1/2
    otherwise
        a = 0.445948490915965; b = 0.108103018168070;
        w1=0.111690794839005; w2=0.054975871827661;
        qp = [a, a;
              a, 1-2*a;
              1-2*a, a;
              b, b;
              b, 1-2*b;
              1-2*b, b];
        qw = [w1; w1; w1; w2; w2; w2]; % sum=1/2  (EKSTRA 1/2 YOK!)
end
end
