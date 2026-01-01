function near = triangles_are_near(tri, t1, t2, epsScale)
c1 = tri.cent(t1,:); c2 = tri.cent(t2,:);
d  = norm(c1-c2);

A1 = 0.5*tri.area2(t1);
A2 = 0.5*tri.area2(t2);
l1 = sqrt(4*A1/sqrt(3));   % eşkenar eşdeğer kenar
l2 = sqrt(4*A2/sqrt(3));
h  = 0.5*(l1+l2);

near = (d < epsScale*h);
end
