#HS48
f(x) =  (x[1] - 1)^2 + (x[2] - x[3])^2 + (x[4] - x[5])^2
x0 = [3.0; 5.0; -3.0; 2.0; -2]
c(x) = [x[1] + x[2] + x[3] + x[4] + x[5] - 5; x[3] - 2*(x[4] + x[5]) + 3;0;0;0]

lcon = [-Inf, -Inf,-Inf,-Inf,-Inf]
ucon = [Inf,Inf,Inf,Inf,Inf]

lvar = [-Inf ; -Inf; -Inf; -Inf; -Inf]
uvar = [ Inf ;  Inf;  Inf;  Inf;  Inf]

nlp = ADNLPModel(x->f(x), x0; c=c, lvar=lvar, uvar=uvar, lcon=lcon, ucon=ucon)
xk, iter, fx, optimal, gradLA, contraintes, yk=Augmented_Lagrangian_Practical(nlp)
nevalobj=neval_obj(nlp)
nevalhess=neval_hess(nlp)
nevaljac=neval_jac(nlp)


@printf("%5i \t %5i \t %5i \t %5i \t %10.10f", nevalobj, nevalhess, nevaljac, iter, fx )