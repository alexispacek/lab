# Autores: Alexis Pacek, Horacio Tettamanti
# Laboratorio 7

# librerias

from scipy.optimize import minimize
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import seaborn as sns
import math as math
from scipy.stats import chisquare
import sympy as sp
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
"""
Se busca simular una lluvia cosmica donde su centro energetico impacta sobre un arreglo hexagonal de detectores.

implementamos un hexagono conformado por 7 detectores (6 perifericos + 1 central).

Randomizamos el centro de la lluvia de modo de agregar incerteza a la posicion del core.

Los detectores miden la intensidad de la lluvia en unidades de VEM siguiendo una ley de potencias de la forma:

    I = s0*(r/r0)**(-b) 

    donde r es la distancia del core al detector, r0 es un parametro para adimensionalizar y s0 y b son parametros de la funcion de potencia.

Nuevamente randomizamos la señal de modo de agregar incerteza a esta.

Obtenemos el baricentro de la señal y su error.

Obtenemos el chi2 de la señal.

Minimizamos el chi2 para obtener los parametros de la funcion de potencia.

    s0 = theta[0] #parametro de la funcion power law
    b = theta[1] #parametro de la funcion power law
    x_core = theta[2] #posicion del core en x
    y_core = theta[3] #posicion del core en y

Ajustamos la funcion de potencia con los parametros obtenidos.

Calculamos el error relativo de la varianza de la señal.

Encontramos el r que optimiza el error relativo

Hacemos un Monte Carlo para obtener la distribucion del r optimo. 

"""
# Parametros

silent = 2
saturacion = 1025

# Defino parametros de LDF (ley de potencias)

r0 = 300  # metros
s0 = 100  # VEM  Vertical Equivalent Muon
angulo = 0  # radianes
b = 2.1  # parametro de beta de la power law


def powerlaw(r, s0, b):  # defino power law para modelar el decaimiento de la señal en el detector
    # r0 no es variable aleatoria
    r0 = 300  # Unidades [m]
    return s0*(r/r0)**(-b)  # unidades [VEM] Vertical Muon Equivalent


# estimacion inicial del core es el baricentro

# Funcion que calcula el baricentro de la señal
def baricentro(signal, det_pos, silent=2, saturacion=1025):
    sum_s = 0  # inicializo la suma de la señal
    bari_x = 0  # inicializo el baricentro en x
    bari_y = 0  # inicializo el baricentro en y
    for i in range(len(signal)):  # para cada punto de la señal
        if signal[i] > silent:  # si la señal es mayor a la silent
            if signal[i] < saturacion:  # si la señal es menor que la saturacion
                # se suma el valor de la señal por la posicion del detector en x
                bari_x += signal[i]*det_pos[i][0]
                # se suma el valor de la señal por la posicion del detector en y
                bari_y += signal[i]*det_pos[i][1]
                sum_s += signal[i]  # se suma el valor de la señal
    return(bari_x/sum_s, bari_y/sum_s)


# funcion que calcula el chi2
def funcion_costo2(theta, x_pos, y_pos, signal_sim, zenith, azimuth, ysigma):

    s0 = theta[0]  # parametro de la funcion power law
    b = theta[1]  # parametro de la funcion power law
    x_core = theta[2]  # posicion del core en x
    y_core = theta[3]  # posicion del core en y

    costo = np.zeros_like(s0)  # inicializo el costo

    for (dist_x, dist_y, y1, sigma1) in zip(x_pos, y_pos, signal_sim, ysigma):  # para cada punto de la señal

        dist = getDistances(dist_x, dist_y, x_core, y_core, zenith, azimuth)[
            0]  # vector de distancias de los detectores respecto al core
        yfit = powerlaw(dist, s0, b)  # vector de señales modeladas
        residuo = y1 - yfit  # vector de residuos
        z = (residuo / sigma1)
        costo += z*z  # se suma el costo

    return costo  # retorno el costo


# costo2 = lambda theta: funcion_costo2(theta,x_det,y_det,datos['signal'],zenith,azimuth,datos['sigma']) #tomo como variable muda theta

def in_hex(poli):  # funcion que determina si un punto esta dentro de un hexagono
    path = mpltPath.Path(poli)  # creo el poligono
    hex = False  # inicializo la variable
    # genero un punto aleatorio
    random_point = (np.random.uniform(-433, 433), np.random.uniform(-375, 375))
    while hex != True:  # mientras no este dentro del hexagono
        # si esta dentro del hexagono
        if path.contains_points([random_point]) == True:
            hex = True  # cambio la variable
        else:  # si no esta dentro del hexagono
            # genero un punto aleatorio
            random_point = (np.random.uniform(-433, 433),
                            np.random.uniform(-375, 375))
    return random_point  # retorno el punto aleatorio


# funcion que calcula las distancias de los detectores respecto al core
def getDistances(xpositions, ypositions, x0=0, y0=0, zenithRad=0, azimuthRad=0):

    # distancias en x #esta es la x que necesito en la formula de la varianza?
    dx = xpositions - x0
    dy = ypositions - y0  # distancias en y

    # distancia del core a los detectores
    groundDistances = np.sqrt(dx*dx+dy*dy)

    # Direction cosines in the plane xy
    dirx = math.cos(azimuthRad)  # direccion en x
    diry = math.sin(azimuthRad)  # direccion en y

    # proyeccion del core en el plano xy
    axisProjections = math.sin(zenithRad) * (dx*dirx+dy*diry)

    # distancia del core a los detectores
    distances = np.sqrt(groundDistances**2 - axisProjections**2)

    # retorno las distancias
    return (distances, axisProjections, groundDistances)


def simZenith(zenithMinDeg, zenithMaxDeg):  # funcion que genera una zenith aleatoria

    # convierto la zenith minima a radianes
    zenithMinRad = math.radians(zenithMinDeg)
    # calculo el seno al cuadrado de la zenith minima
    sin2MinZenith = math.sin(zenithMinRad)**2

    # convierto la zenith maxima a radianes
    zenithMaxRad = math.radians(zenithMaxDeg)
    # calculo el seno al cuadrado de la zenith maxima
    sin2MaxZenith = math.sin(zenithMaxRad)**2

    rng = np.random.default_rng()  # genero un generador de numeros aleatorios
    u = rng.random()  # genero un numero aleatorio

    sin2Zenith = sin2MinZenith + u * (sin2MaxZenith-sin2MinZenith)
    sinZenith = math.sqrt(sin2Zenith)

    zenith = math.asin(sinZenith)
    azimut = np.random.uniform(0, 2*np.pi)

    return (zenith, azimut)  # retorno la zenith y el azimut

# Geometria del arreglo de los detectores


poligono = [[-216.5, -375], [216.5, -375], [433, 0], [216.5, 375],
            [-216.5, 375], [-433, 0]]  # poligono que contiene el hexagono
detectores_pos = [[-216.5, -375], [216.5, -375], [433, 0], [216.5, 375],
                  [-216.5, 375], [-433, 0], [0, 0]]  # posiciones de los detectores
rectangulo = [[-433, -375], [433, -433], [433, 375],
              [-433, 375]]  # poligono que contiene el rectangulo
pointsx, pointsy = [-216.5, 216.5, 433, 216.5, -216.5, -433.5,
                    0], [-375, -375, 0, 375, 375, 0, 0]  # posiciones de los puntos
centro = (0, 0)  # detector central y centro del hexagono
# inicializo los vectores de x de los detectores
x_det = np.zeros(len(detectores_pos))
# inicializo los vectores de y de los detectores
y_det = np.zeros(len(detectores_pos))
for i in range(len(detectores_pos)):  # para cada detector
    x_det[i] = detectores_pos[i][0]  # tomo la posicion en x
    y_det[i] = detectores_pos[i][1]  # tomo la posicion en y

# Defino parametros de LDF

zenith = simZenith(-angulo, angulo)[0]  # zenith
azimuth = simZenith(-angulo, angulo)[1]  # azimuth


# defino la función evento:
# genera un rayo cosmico con una zenith y un azimut fijo
# randomiza la señal y devuelve como output un plot de la LDF, Varianza vs distancia y el R optimo.

def evento(plots):

    # Genero el core de impacto de la lluvia
    random_point = in_hex(poligono)

    # calculo las distancias entre el evento y los 7 detectores.
    distancias = getDistances(np.asarray(pointsx), np.asarray(
        pointsy), random_point[0], random_point[1], simZenith(-angulo, angulo)[0], simZenith(-angulo, angulo)[1])
    r = np.asarray(distancias[0])

    b_real = 2.1
    # Evaluo la señal
    signal = powerlaw(r, s0, b_real)
    sigma_signal = np.sqrt(signal)  # error poissoniano

    # Randomizo la señal
    y_random = np.random.normal(
        signal, sigma_signal, size=len(signal))  # señal simulada

    # diccionario con indices como r y señal y sigma
    datos = {'r': r, 'signal': y_random, 'sigma_signal': sigma_signal}

    # Impongo filtro
    # los tanques no miden señales menores a 2 VEM y saturan en un 95% de las mediciones en 1025VEM

    datos = pd.DataFrame(
        datos, columns=['r', 'signal', 'sigma_signal'], index=r, dtype=float)

    # Defino el baricentro para usarlo como la semilla inicial para el ajuste

    seed_x = baricentro(np.array(datos['signal']), detectores_pos)[0]
    seed_y = baricentro(np.array(datos['signal']), detectores_pos)[1]

    # print('Semilla:',seed_x,seed_y)

    # Inicio la minimizacion
    def costo2(theta): return funcion_costo2(theta, x_det, y_det,
                                             datos['signal'], zenith, azimuth, datos['sigma_signal'])  # tomo como variable muda theta

    # obtengo los parametros estimados s0, b, x_core, y_core
    res = minimize(costo2, x0=(30, 2, seed_x, seed_y), tol=1e-4)
    cova = 2*res.hess_inv

    s0_est = res.x[0]
    b_est = res.x[1]
    x_core_est = res.x[2]
    y_core_est = res.x[3]

    # obtengo la distancia usando getDistances con x_core_est y y_core_est

    distancias_est = getDistances(np.asarray(pointsx), np.asarray(
        pointsy), x_core_est, y_core_est, simZenith(-angulo, angulo)[0], simZenith(-angulo, angulo)[1])
    r_est = np.asarray(distancias_est[0])
    datos['r_est'] = r_est

    filtro = (datos['signal'] > silent) & (datos['signal'] < saturacion)
    datos = datos[filtro]

    # si se exitaron menos o igual que 4 detectores entonces no se puede ajustar
    if len(datos) <= 4:
        print('No se puede ajustar')
        return None, None, None

    if res.success == False:

        return None, None, None

    r = np.linspace(min(r_est), max(r_est), 1000)

    b = sp.symbols('b')
    s = sp.symbols('s')
    R = sp.symbols('R')
    r0 = 300
    S = sp.Function('S')
    S = s*(R/r0)**(-b)  # power law
    # S = s*(R/r0)**(-b)*(1 + R/r0)**(-b) #NKG

    # derivada simbolica de S con respecto a s
    deri_s0 = sp.diff(S, s)
    derivada_s0 = []
    for u in r:
        derivada_s0.append(deri_s0.subs([(R, u), (b, b_est), (s, s0_est)]))

    # derivada simbolica de S con respecto a s
    deri_s0 = sp.diff(S, s)
    derivada_s0 = []
    for u in r:
        derivada_s0.append(deri_s0.subs([(R, u), (b, b_est), (s, s0_est)]))

    # derivada simbolica de S con respecto a b
    deri_b = sp.diff(S, b)
    derivada_b = []
    for u in r:

        derivada_b.append(deri_b.subs([(R, u), (b, b_est), (s, s0_est)]))

    derivada_s0 = np.array(derivada_s0)
    derivada_b = np.array(derivada_b)

    sigma_s0 = np.sqrt(cova[0, 0])
    sigma_b = np.sqrt(cova[1, 1])

    cov_s_b = cova[0, 1]

    var_mu_est = (derivada_s0*sigma_s0)**2 + (derivada_b *
                                              sigma_b)**2 + 2*derivada_s0*derivada_b*cov_s_b

    sigma_mu_est = var_mu_est**(0.5)
    sigma_mu_est = np.array(sigma_mu_est, dtype=float)

    error_relativo = sigma_mu_est/powerlaw(r, s0_est, b_est)
    error_rela_dic = {'R': r, 'error_relativo': error_relativo}
    # error_rela_dic to dataframe
    error_rela_df = pd.DataFrame(error_rela_dic)

    bias_s0 = (s0_est - s0)/s0
    bias_b = (b_est - b_real)/b_real

    minimo = error_rela_df.loc[error_rela_df['error_relativo'].idxmin()][0]
    if plots == True:
        # grafico la señal estimada con su ajuste power law
        plt.figure(figsize=(11, 11))

        #ax1.scatter(r_est,datos['signal'],color='blue',label='Señal estimada',linestyle='dotted')
        plt.errorbar(datos['r_est'], datos['signal'], yerr=datos['sigma_signal'],
                     color='red', label='Señal estimada con error', fmt='o')
        b_round = round(b_est, 2)
        s0_round = round(s0_est, 2)
        plt.plot(r, powerlaw(r, s0_est, b_est), color='orange',
                 label='Ajuste power law', linestyle='dotted')
        #plt.errorbar(R, powerlaw(R,s0_est,b_est), yerr = sigma_mu_est,label='error', alpha=0.1)
        plt.fill_between(r, powerlaw(r, s0_est, b_est)-sigma_mu_est,
                         powerlaw(r, s0_est, b_est)+sigma_mu_est, color='tab:orange', alpha=0.2)

        plt.xlabel('Distancia [m]', fontsize=14)
        plt.ylabel('Señal [VEM]', fontsize=15)
        plt.title('Señal estimada con ajuste power law', fontsize=20)
        plt.legend(fontsize=15)
        plt.grid()
        plt.tick_params(axis='y', labelsize=13)
        plt.tick_params(axis='x', labelsize=13)
        plt.show()

        plt.figure(figsize=(11, 11))
        plt.plot(r, error_relativo, color='tab:orange', label='Error relativo')
        plt.xlabel('Distancia [m]', fontsize=15)
        plt.ylabel('Error relativo', fontsize=15)
        plt.title('Error relativo', fontsize=20)
        plt.axvline(x=minimo, color='black', linestyle='dotted',
                    label=r'$r_{opt} =$'+str(round(minimo))+' m')
        plt.legend(fontsize=15)
        plt.grid()
        # agrego al eje x el valor de minimo

        plt.tick_params(axis='y', labelsize=13)
        plt.tick_params(axis='x', labelsize=13)
        plt.show()

        plt.figure(figsize=(8, 8))
        plt.xlabel('[m]', fontsize=15)
        plt.ylabel('[m]', fontsize=15)
        plt.scatter(pointsx, pointsy, label='Detectores')
        plt.scatter(seed_x, seed_y, c='black', label='Baricentro')
        plt.scatter(random_point[0], random_point[1],
                    s=90, c='green', label='Core simulado')
        plt.scatter(res.x[2], res.x[3], s=90, c='red', label='Core ajustado')
        plt.tick_params(axis='y', labelsize=13)
        plt.tick_params(axis='x', labelsize=13)
        plt.legend(fontsize=13)
        #plt.text(x=seed_x, y=seed_y, s='Baricentro',size = 15)
        #plt.text(x=res.x[2],y=res.x[3], s='Core estimado',size = 15)
        #plt.text(x=random_point[0], y=random_point[1], s='Core simulado',size = 15)
        plt.grid()
        plt.show()

    return minimo, bias_s0, bias_b

# Monte Carlo para obtener r_optimo y los bias de los estimadores


optimos = []
bi_s0 = []
bi_b = []
for i in range(0, 100):
    optimo, bias_S, bias_B = evento(False)
    optimos.append(optimo)
    bi_s0.append(bias_S)
    bi_b.append(bias_B)
    # ploteo la distribución del r_optimo
# elimino nones de optimos
optimos = [x for x in optimos if x is not None]
bi_b = [x for x in bi_b if x is not None]
bi_s0 = [x for x in bi_s0 if x is not None]


# ploteo de la distribución de r_optimo

std = np.std(optimos)/np.sqrt(len(optimos))
mean = np.mean(optimos)


plt.figure(figsize=(11, 11))
plt.hist(optimos, bins=50, density=False)
plt.xlabel('Distancia [m]', fontsize=20)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.ylabel('Número de eventos', fontsize=20)
plt.axvline(x=np.mean(optimos), color='r', linestyle='--', linewidth=2)
plt.text(x=mean+10, y=300,
         s=r'$r_{opt}:$ '+str(round(mean)) + r'$\pm$'+str(round(std))+' m', size=20)
plt.title('Distribución de la distancia óptima ', fontsize=20)
plt.xlim(200, 600)
plt.grid(False)
plt.show()


# formato seaborn

#plots in line
# %matplotlib qt

# sesgo de s0
plt.figure(figsize=(11, 11))
plt.hist(bi_s0, bins=20, density=False)
plt.axvline(x=np.mean(bi_s0), color='r', linestyle='--')
plt.text(x=np.mean(bi_s0)+0.04, y=140, s='Sesgo: '+str(round(np.mean(bi_s0), 3)
                                                       ) + r'$\pm$'+str(round(np.std(bi_s0)/np.sqrt(len(bi_s0)), 3)), size=20)
plt.xlabel(r'$(S_{rec}-S_{sim})/S_{sim}$', size=20)
plt.ylabel('Número de eventos', fontsize=20)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.title(r'Distribución del parametro $S_{0}$', fontsize=20)
plt.xlim(-0.35, 0.35)
plt.grid(False)
plt.show()


# sesgo de b
plt.figure(figsize=(11, 11))
plt.hist(bi_b, bins=20, density=False)
plt.tick_params(axis='y', labelsize=15)
plt.axvline(x=np.mean(bi_b), color='r', linestyle='--')
plt.text(x=np.mean(bi_b)+0.05, y=120, s='Sesgo: '+str(round(np.mean(bi_b), 3)
                                                      )+r'$\pm$'+str(round(np.std(bi_b)/np.sqrt(len(bi_b)), 3)), size=20)
plt.tick_params(axis='x', labelsize=15)
plt.xlabel(r'$(S_{rec}-S_{sim})/S_{sim}$', fontsize=20)
plt.ylabel('Número de eventos', fontsize=20)
plt.title(
    'Distribución del parametro \N{GREEK SMALL LETTER BETA}', fontsize=20)
plt.xlim(-0.4, 0.4)
plt.grid(False)
plt.show()
