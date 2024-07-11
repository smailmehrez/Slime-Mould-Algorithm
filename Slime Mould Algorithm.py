from Fonction import*
import numpy as np
import random
import matplotlib.pyplot as plt
import math




import warnings
warnings.filterwarnings('ignore')


def SMA(func,Solution,nb_solu,taille_solu,itération):
    T = itération
    taille_solution = taille_solu
    nb_solution = nb_solu
    Matrice_Solution =np.copy(Solution)

    #********************SMA prinssipale*******************************
    idxMin = np.argmin(Matrice_Solution[:,taille_solution])
    idxMax = np.argmax(Matrice_Solution[:,taille_solution])
    solution_best_in_itra=np.copy(Matrice_Solution[idxMin,:])
    vecteur_Fonc_objectiv_meillure=np.full(shape=(T+1), fill_value=Matrice_Solution[idxMin,taille_solution],dtype=float)
    _,UB,_,LB=func(([1,2]))
    # ********* Inialiser les parametre utilise dans algorithme SMA **********************
    t = 1
    Z=0.1
    while t <= T :
        for i in range(nb_solution):
            r1=random.uniform(0,1)              # initialiser r1
            if Z >= r1:
                # **** phase d'exploration *************************
                # Mettre à jour la solution 𝑺𝒊𝒕
                r2 = random.uniform(0,1)   # initialiser r2
                Matrice_Solution[i,:-1] = LB + r2 * (UB - LB)
            else:
                # ******* phase d'exploitation *******************
                # Calculer 𝒑𝒊
                Sx = math.fabs(Matrice_Solution[i,taille_solution]-solution_best_in_itra[taille_solution])# |f(Si) - f(S*)|
                pi = math.tanh(Sx)  # 𝑡𝑎𝑛ℎ|𝑓(𝑆𝑖)−𝑓(𝑆∗)|
                r3 = random.uniform(0,1)  # initialiser r3
                if pi <= r3 :
                    # calcule Vc
                    b=1 -(t / T)  # initialiser b
                    Vc = random.uniform(-b,b)
                    Matrice_Solution[i,:-1] = Vc * Matrice_Solution[i,:-1] # St+1=Vc*St

                else:
                    gi = ((Matrice_Solution[idxMin,taille_solution]-Matrice_Solution[i,taille_solution])/(Matrice_Solution[idxMin,taille_solution]-Matrice_Solution[idxMax,taille_solution]))+1
                    r = random.uniform(0, 1) # initialiser r
                    if i < (nb_solution/2):
                        Wi = 1 + r * float(math.log(math.fabs(gi))) # Wi=1+rlog(gi)   
                    else:
                        Wi = 1 - r * float(math.log(math.fabs(gi))) # Wi=1-rlog(gi)
                    a = math.atan(1-(t/T))   # initialiser a
                    # ***** selection deux vecteur de facon aleatoire **
                    SA = random.randint(0, nb_solution - 1)  # initialiser Sa
                    SB = random.randint(0, nb_solution - 1)  # initialiser Sb
                    while SA == SB:
                        SB = random.randint(0, nb_solution - 1)  # si Sa = Sb
                    Vb = random.uniform(-a,a)   # initialiser Vb
                    Matrice_Solution[i,:-1] = solution_best_in_itra[:-1] + Vb * ((Wi*Matrice_Solution[SA,:-1]) - Matrice_Solution[SB,:-1]) # St+1=S* +Vb*(Wi*Sa-Sb)

            # *********************traiter les borne******************************************************
            for j in range(taille_solution):
                if Matrice_Solution[i,j] < LB:
                    Matrice_Solution[i,j] = "{0:.3f}".format(random.uniform(LB,0))
                if Matrice_Solution[i,j] > UB:
                    Matrice_Solution[i,j] = "{0:.3f}".format(random.uniform(0,UB))
            # ***********************calcule la fonction objective***************
            Matrice_Solution[i,taille_solution],_,_,_=func(Matrice_Solution[i,:-1])

            # *********************Fin de traitment SMA**************************
        idxMin = np.argmin(Matrice_Solution[:,taille_solution])
        idxMax = np.argmax(Matrice_Solution[:,taille_solution])
        if (solution_best_in_itra[taille_solution]> Matrice_Solution[idxMin,taille_solution]):
            solution_best_in_itra=np.copy(Matrice_Solution[idxMin,:])
        vecteur_Fonc_objectiv_meillure[t] = solution_best_in_itra[taille_solution]
        
        t = t + 1
    return(vecteur_Fonc_objectiv_meillure)



T=100
n=10
fonction=Quartic
_,UB,taille_solution,LB=fonction(([1,2]))
nb_solution=n
# crer matrice du Solution
M1 = np.full(shape=(nb_solution, taille_solution+1), fill_value=0,dtype=np.float64)
# inialiser le Matrice du solution random
for j in range(nb_solution):
    for i in range(taille_solution):
        M1[j,i] = "{0:.3f}".format(random.uniform(LB, UB)) 
# calcule la fonction objective 
for j in range(nb_solution):
    M1[j,taille_solution],_,_,_=fonction(M1[j,:-1])
vecteur_best=SMA(fonction,M1,nb_solution,taille_solution,T)
ax1 =plt.axes()
ax1.plot(vecteur_best,label="SMA",c="#4B2991")
ax1.set_ylabel('Coût F')
ax1.set_xlabel("Nombre d'itération")
if fonction== Ackley or fonction == Michalewicz or fonction== Schwefel1 or fonction== Schwefel2_26 or fonction== Himmelblau:
    ax1.legend()
    ax1.set_title("Le Coût f en fonction de Nombre d'itération")
    plt.show()
else:
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title("Le Coût f en fonction de Nombre d'itération")
    plt.show()