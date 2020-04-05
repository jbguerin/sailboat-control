import socket
import select
from sailboat_github import *

hote = 'localhost'
port = 12800
connexion_principale = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # Création de la socket
connexion_principale.bind((hote, port))
connexion_principale.listen(5)                                          # En écoute
print("Le serveur écoute à présent sur le port {}".format(port))

serveur_lance = True
clients_connectes = []

# Variable pour la simulation
dt = 0.1
awind, ψ = 50, pi/2
listex, listey = [], []
x = array([[0, 0, -3, 3, 0]]).T  # x=(x,y,θ,v,w)
ax=init_figure(-100,100,-60,60)

client = False
clients_a_lire = []

while client == False:
    connexions_demandees, wlist, xlist = select.select([connexion_principale], [], [], 0.025)
    for connexion in connexions_demandees:
        connexion_avec_client, infos_connexion = connexion.accept()
        clients_connectes.append(connexion_avec_client)
        client = True

while len(clients_a_lire)==0:
    clients_a_lire, wlist, xlist = select.select(clients_connectes, [], [], 0.025)

commande = 0
while serveur_lance:
    for client in clients_a_lire:
        # Client est de type socket
        msg_recu = client.recv(1024)

        msg_recu = int(msg_recu.decode('utf8')[-1])
        print("Reçu :", msg_recu)
        if msg_recu != "fin":

            clear(ax)
            update_ax(x,ax, commande)

            a = array([[x[0][0]], [x[1][0]]])
            b = array([[x[0][0] + cos(x[2][0])],
                        [x[1][0] + sin(x[2][0])]])

            if msg_recu == 2:
                commande = 1
            elif msg_recu == 1:
                commande = -1
            elif msg_recu == 3:
                commande = 0

            listex.append(a[0, 0]), listex.append(b[0, 0])
            listey.append(a[1, 0]), listey.append(b[1, 0])
            plot(listex, listey, 'blue')  # afficher la trace complète au fur et à mesure (tous les segments)

            u = control(x, a, b, commande)

            xdot, δs = f(x, u)
            x = x + dt * xdot
            draw_sailboat(x, δs, u[0, 0], ψ, awind)

print("Fermeture des connexions")
for client in clients_connectes:
    client.close()

connexion_principale.close()
