function subTris = subdivide_triangles(A,B,C, depth)
subTris = cat(3, [A;B;C]);
for d=1:depth
    old = subTris;
    Ns0 = size(old,3);
    new = zeros(3,3,4*Ns0);
    t=1;
    for s=1:Ns0
        A0=old(1,:,s); B0=old(2,:,s); C0=old(3,:,s);
        AB=0.5*(A0+B0);
        BC=0.5*(B0+C0);
        CA=0.5*(C0+A0);
        new(:,:,t)=[A0;AB;CA]; t=t+1;
        new(:,:,t)=[AB;B0;BC]; t=t+1;
        new(:,:,t)=[CA;BC;C0]; t=t+1;
        new(:,:,t)=[AB;BC;CA]; t=t+1;
    end
    subTris = new;
end
end
