function r = map_ref_to_tri(tri, t, qp)
A = tri.r1(t,:); B = tri.r2(t,:); C = tri.r3(t,:);
u = qp(:,1); v = qp(:,2);
r = A + u.*(B-A) + v.*(C-A);
end
