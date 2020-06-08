using NLPModels
using LinearAlgebra
using JSOSolvers
using Printf
using OptimizationProblems
using NLPModelsJuMP
using JuMP


#Mathieu Gervais-Dubé 1841376, Charles Brosseau 1804072

#Package NLPModels et JSOSolvers de Abel Soares Siqueira & Dominique Orban

#NLPModels: Abel Soares Siqueira, & Dominique Orban. (2019, February 6). NLPModels.jl. Zenodo.
#http://doi.org/10.5281/zenodo.2558627

#Code de la méthode inspiré de JSOSolvers (Abel Soares Siqueira, & Dominique Orban.)
#Dans le cadre d'un cours donné par Dominique Orban.

#Methode du lagrangien augmenté

function Augmented_Lagrangian_Practical(nlp:: AbstractNLPModel;
                              xk :: AbstractVector=copy(nlp.meta.x0),
                              yk :: AbstractVector=copy(nlp.meta.x0),
                              pk :: Real=10,
                              KKT_atol :: Real=1*10^(-7),
                              KKT_rtol :: Real=1*10^(-6),
                              nm_itmax :: Int=100)
    #Contraintes
    c(x)=cons(nlp,x)
        
        
    contraintes_0=c(xk)
    contraintes= contraintes_0;
    norm_c0=norm(contraintes_0);
    norm_c=norm_c0;
    
    #Gradient, Jacobienne
    g=grad(nlp,xk)
    J=jac(nlp,xk)
    gradLA=g-Transpose(J)*yk
    gradLA_0=gradLA;
    normGrad=norm(gradLA);
    normGrad0=normGrad;
    
    #Initialisaton de f(x)
    f(x)=obj(nlp,x);
    fx=-1;

    #Initialisation de variables utiles
    optimal=false;
    fin=false;
    iter=0;
    pk=10
    omega_k=1/pk;
    eta_k=1/((pk)^(0.1));
        
    #Affichage des informations du tableau
    @printf("%s", "  iter |") 
    for i in 1:size(xk,1)
           @printf("%s %1.0f %s","\t x",i,"  \t|")
    end
    @printf("%s", "      f(xk)       |        ||c||      |  ||gradLA||   |\n")
    
    while(!fin)
         #Affichage des valeurs à chaque itération
        @printf("%3d\t", iter) 
        for i in 1:size(xk,1)
           @printf("%10.10f\t", xk[i] )
        end
        @printf("%10.10f\t  %10.10f\t  %10.10f\t \n", f(xk), norm_c, normGrad)
        
        #Modification de la fonction pour la rendre sans contraintes
        L(x)=f(x)-Transpose(yk)*c(x)+0.5*pk*norm(c(x))^2 #Modifié pour c(x)^2
 
        #Création du nouveau modèle sans contraintes à optimiser
        nlp_unconstrained=ADNLPModel(x->L(x), xk, lvar=nlp.meta.lvar, uvar=nlp.meta.uvar);
        result=tron(nlp_unconstrained, atol=omega_k, rtol=0,max_eval=1000);             
        @printf("%s  \n \n","ATTENTION! tron n'a pas pu résoudre le sous-problème")
            

        xk=result.solution;
        
       
        
        contraintes=c(xk);
        norm_c=norm(contraintes)
        
        g=grad(nlp,xk);
        J=jac(nlp,xk);
        gradLA=g-Transpose(J)*(yk);
        normGrad=norm(gradLA)
        
        if norm_c<=eta_k
            #Mise à jour des multiplicateurs
            yk=yk-pk*contraintes;
            
            #Test KKT pour optimalité
            optimal1=all(norm_c<=KKT_atol+KKT_rtol*norm_c0);
            optimal2=normGrad<=KKT_atol+KKT_rtol*normGrad0;
            optimal=optimal1 && optimal2;
            
            #Arrêt si condition d'optimalité atteintes ou itérations atteintes
            if optimal
                fx=f(xk);
                fin=true;
                @printf("%s  \n \n","Solution trouvée (first order) at xk: ")
                println(xk);
                break;
            end
            if iter>=nm_itmax 
                fx=f(xk);
                fin=true;
                @printf("%s  \n \n","Timeout (iter>=itermax) ")
                break;
            end   
         
        eta_k=eta_k/((pk).^(0.9));
        omega_k=omega_k/pk;
 
        else
            
        pk=100*pk;    
        eta_k=eta_k/((pk).^(0.1));
        omega_k=1/pk;

        end
        iter=iter+1;
    
    end
    
    return xk, iter, fx, optimal, gradLA, contraintes, yk

end
