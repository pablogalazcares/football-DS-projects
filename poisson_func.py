import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
from tqdm import tqdm
from scipy.stats import poisson, skellam
from scipy.optimize import minimize
import statsmodels.api as sm
import statsmodels.formula.api as smf
from poisson_func import * 

plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (16,9) 


def poisson_tournament(df_matches, poisson_model, N):
    inicio = datetime.now()

    df_match = df_matches[df_matches.Jugado == 0]

    index = range(df_match.shape[0])
    columns = range(1,N+1)

    sim_poisson_local = pd.DataFrame(index=index, columns=columns)
    sim_poisson_visita = pd.DataFrame(index=index, columns=columns)
    sim_poisson_local['idMatch'] = df_match.index
    sim_poisson_local['Local'] = df_match.Local.values
    sim_poisson_visita['idMatch'] = df_match.index
    sim_poisson_visita['Visita'] = df_match.Visita.values

    sim_poisson_local.set_index(['idMatch',"Local"], inplace = True)
    sim_poisson_visita.set_index(['idMatch',"Visita"], inplace = True)

    for i, id_match in tqdm(enumerate(df_match.index)):
        local = df_match.iloc[i].Local
        away = df_match.iloc[i].Visita
        lambda_local = poisson_model.predict(pd.DataFrame(data={'Equipo':local, 'Rival':away, 'Localia':1}, index = [1]))
        lambda_away = poisson_model.predict(pd.DataFrame(data={'Equipo':away, 'Rival':local, 'Localia':0}, index = [1]))
        goles_local = np.random.poisson(lambda_local, N)
        goles_visita = np.random.poisson(lambda_away, N)

        sim_poisson_local.iloc[i] = goles_local
        sim_poisson_visita.iloc[i] = goles_visita

    sim_poisson_local.to_csv("sim_poisson_local.csv")
    sim_poisson_visita.to_csv("sim_poisson_visita.csv")
    final = datetime.now()
    print(final-inicio)
    return sim_poisson_local, sim_poisson_visita

def current_table(df, teams):
    columns = ["Posición","Equipo","PJ","Puntos","DG","GF","GC"]
    df_table = pd.DataFrame(index = range(len(teams)), columns = columns)
    for t,team in enumerate(teams):
        rend_local = df[(df.Local == team)&(df.Jugado == 1)]
        rend_visita = df[(df.Visita == team)&(df.Jugado == 1)]

        pts_local = len(rend_local[rend_local.GL > rend_local.GV])*3 + len(rend_local[rend_local.GL == rend_local.GV])
        pts_visita = len(rend_visita[rend_visita.GL < rend_visita.GV])*3 + len(rend_visita[rend_visita.GL == rend_visita.GV])
        pts = pts_local + pts_visita

        goles_favor = rend_local.GL.sum() + rend_visita.GV.sum()
        goles_contra = rend_local.GV.sum() + rend_visita.GL.sum()
        dif_goles = goles_favor - goles_contra

        pj = df[((df.Local == team) | (df.Visita == team))&(df.Jugado == 1)].shape[0]
        df_table.at[t] = [0, team, pj, pts, dif_goles, goles_favor, goles_contra]
     
    df_table.sort_values(by=["Puntos","DG","GF","GC"], inplace = True, ascending = False)
    df_table["Posición"] = range(1,19)
    df_table.set_index("Equipo", inplace = True)
    df_table.to_csv("Tabla2020.csv")
    return df_table

def summary_positions(sim_poisson_local, sim_poisson_visita, N_sim, teams, df_tabla_2019, df_tabla_2020):
    #team, n_sim, posición
    team_stats = []
    for n_sim in tqdm(range(1,N_sim+1)):
        df_table_sim = df_tabla_2020.copy()
        for team in teams:
            #info partidos de local
            a = sim_poisson_local[sim_poisson_local.index.get_level_values("Local") == team].reset_index().drop("Local", axis = 1).set_index("idMatch")
            b = sim_poisson_visita[sim_poisson_visita.index.get_level_values("Visita") == team].reset_index().drop("Visita", axis = 1).set_index("idMatch")

            #info partidos de visita
            aa = sim_poisson_visita[sim_poisson_visita.index.get_level_values(0).isin(a.index.get_level_values("idMatch"))].reset_index().drop("Visita", axis = 1).set_index("idMatch")
            bb = sim_poisson_local[sim_poisson_local.index.get_level_values(0).isin(b.index.get_level_values("idMatch"))].reset_index().drop("Local", axis = 1).set_index("idMatch")

            pts = 3*(sum(a[n_sim] > aa[n_sim])) + (sum(a[n_sim] == aa[n_sim])) + 3*(sum(b[n_sim] > bb[n_sim])) + (sum(b[n_sim] == b[n_sim]))
            gf = sum(a[n_sim])
            gc = sum(b[n_sim])

            df_table_sim.loc[team, "Puntos"] = df_table_sim.loc[team, "Puntos"] + pts
            df_table_sim.loc[team, "GF"] = df_table_sim.loc[team, "GF"] + gf
            df_table_sim.loc[team, "GC"] = df_table_sim.loc[team, "GC"] + gc
            df_table_sim["DF"] = df_table_sim.GF - df_table_sim.GC
                
            df_table_sim.sort_values(by=["Puntos","DG","GF","GC"], inplace = True, ascending = False)
            df_table_sim["Posición"] = range(1,18+1)
            
        df_tabla_pond = tabla_pond(teams, df_tabla_2019, df_table_sim)
        for team in teams:
            team_stats.append([team, "Absoluta", n_sim, df_table_sim.loc[team, "Posición"]])
            team_stats.append([team, "Ponderada", n_sim, df_tabla_pond.loc[team, "Posición"]])

    df_posicion = pd.DataFrame(team_stats, columns = ["Equipo","Tabla","n_sim","Posición"])
    df_posicion.to_csv("df_posicion.csv")
    return df_posicion

def tabla_pond(teams, df_tabla_2019, df_tabla_2020):
    pond_19 = 0.6
    pond_20 = 0.4
    pond_team_stats = []
    for team in teams:
        pts_2020 = df_tabla_2020.loc[team]["Puntos"]
        pj_2020 = df_tabla_2020.loc[team]["PJ"]
        if team in ["Santiago Wanderers","La Serena"]:
            score = pts_2020/pj_2020
        else:
            pts_2019 = df_tabla_2019.loc[team]["PTS"]
            pj_2019 = 24
            score = (pts_2019/pj_2019)*0.6 + (pts_2020/pj_2020)*0.4

        pond_team_stats.append([team, pts_2019, pts_2020, score])
    df_tabla_pond = pd.DataFrame(pond_team_stats, columns = ["Equipo","2019","2020","Score"])
    df_tabla_pond.sort_values(by=["Score"], ascending = False, inplace = True)
    df_tabla_pond["Posición"] = range(1,df_tabla_pond.shape[0]+1)
    df_tabla_pond["Score"] = df_tabla_pond.Score.round(3)
    df_tabla_pond.set_index("Equipo", inplace = True)
    return df_tabla_pond


def relegation_match_simulation(team_1, team_2, poisson_model):
    lambda_local = poisson_model.predict(pd.DataFrame(data={'Equipo':team_1, 'Rival':team_2, 'Localia':0}, index = [1]))
    lambda_away = poisson_model.predict(pd.DataFrame(data={'Equipo':team_2, 'Rival':team_1, 'Localia':0}, index = [1]))
    goles_local = np.random.poisson(lambda_local)[0]
    goles_visita = np.random.poisson(lambda_away)[0]
    if goles_local > goles_visita:
        ganador = team_1
    elif goles_local < goles_visita:
        ganador = team_2
    else:
        ganador = np.random.choice([team_1, team_2])
    return ganador

def relegation_stats(N_sim, df_posicion, poisson_model):
    # cuantificar cuántas veces no se juega partido de definición por temas de casos difusos
    desc_directo_1, desc_directo_2, desc_directo_3 = 0, 0, 0
    desc_stats = []
    for n_sim in tqdm(range(1,N_sim+1)):
        # Tabla Absoluta (2020)
        ult_abs = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Absoluta")&
                              (df_posicion.Posición == 18)]["Equipo"].iloc[0]
        pen_abs = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Absoluta")&
                              (df_posicion.Posición == 17)]["Equipo"].iloc[0]
        ant_abs = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Absoluta")&
                              (df_posicion.Posición == 16)]["Equipo"].iloc[0]

        # Tabla Ponderada (2019-2020)
        ult_pon = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Ponderada")&
                              (df_posicion.Posición == 18)]["Equipo"].iloc[0]
        pen_pon = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Ponderada")&
                              (df_posicion.Posición == 17)]["Equipo"].iloc[0]
        ant_pon = df_posicion[(df_posicion.n_sim == n_sim)&(df_posicion.Tabla == "Ponderada")&
                              (df_posicion.Posición == 16)]["Equipo"].iloc[0]

        # primer descendido: último de la absoluta
        desc_1 = ult_abs
        n_desc_1 = "Último Absoluta"
        if ult_abs == ult_pon:
            desc_2 = pen_pon
            n_desc_2 = "Penúltimo Ponderada"
            if pen_abs == pen_pon:
                desc_directo_1 += 1
                if ant_abs == ant_pon:
                    desc_3 = ant_abs
                    n_desc_3 = "Antepenúltimo Ponderada y Absoluta"
                elif ant_abs != ant_pon:
                    desc_3 = relegation_match_simulation(ant_abs, ant_pon, poisson_model)
                    n_desc_3 = "Partido Ant Abs vs Ant Pon"
            else: #pen_abs != pen_pon
                if pen_abs == ant_pon:
                    desc_3 = pen_abs
                    n_desc_3 = "Penúltimo Absoluta"
                else: #pen_abs != ant_pon
                    desc_3 = relegation_match_simulation(pen_abs, ant_pon, poisson_model)
                    n_desc_3 = "Partido Pen Abs vs Ant Pon"
        else: #ult_abs != ult_pon
            desc_2 = ult_pon
            n_desc_2 = "Último Ponderada"
            if ult_abs == pen_pon:
                if pen_abs == ult_pon:
                    desc_directo_2 +=1
                    desc_3 = relegation_match_simulation(ant_abs, ant_pon, poisson_model)
                    n_desc_3 = "Partido Ant Abs vs Ant Pon"
                else: # pen_abs != ult_pon
                    desc_3 = relegation_match_simulation(pen_abs, ant_pon, poisson_model)
                    n_desc_3 = "Partido Pen Abs vs Ant Pon"
            
            else: #ult_abs != pen_pon
                if pen_abs == ult_pon:
                    desc_directo_3 += 1
                    desc_3 = relegation_match_simulation(ant_abs, pen_pon, poisson_model)
                    n_desc_3 = "Partido Ant Abs vs Pen Pon"
                else: #pen_abs != pen_pon
                    desc_3 = relegation_match_simulation(pen_abs, pen_pon, poisson_model)
                    n_desc_3 = "Partido Pen Abs vs Pen Pon"

        descendidos = {desc_1, desc_2, desc_3}
        n_desc = len(descendidos)
        if n_desc != 3:
            print("Error de casos")
            print("----------")
            print(ult_abs, pen_abs, ant_abs)
            print(ult_pon, pen_pon, ant_pon)
            print(n_sim, desc_1, "--", n_desc_1)
            print(n_sim, desc_2, "--", n_desc_2)
            print(n_sim, desc_3, "--", n_desc_3)
        else:
            desc_stats.append([n_sim, desc_1, "1", n_desc_1])
            desc_stats.append([n_sim, desc_2, "2", n_desc_2])
            desc_stats.append([n_sim, desc_3, "3", n_desc_3])
        df_desc_stats = pd.DataFrame(desc_stats, columns = ["n_sim","Equipo","Desc","Motivo"])
    
    print(desc_directo_1, desc_directo_2, desc_directo_3)
    df_desc_stats.to_csv("df_desc_stats.csv")
    return df_desc_stats

def cases_distribution(df_posicion, N_sim):
    df_last = df_posicion[df_posicion["Posición"].isin([16,17,18])]

    l1a, l1b = 0, 0
    l2a, l2b, l2c, l2d = 0, 0, 0, 0
    l3a, l3b, l3c, l3d, l3e, l3f = 0, 0, 0, 0, 0, 0

    matches_cd3 = []
    relegation_matches = []

    for n_sim in tqdm(df_last.n_sim.unique()):
        df_sim = df_last[(df_last.n_sim == n_sim)]
        # Absoluta
        A = df_sim[(df_sim.Tabla == "Absoluta")&(df_sim["Posición"] == 18)]["Equipo"].iloc[0]
        B = df_sim[(df_sim.Tabla == "Absoluta")&(df_sim["Posición"] == 17)]["Equipo"].iloc[0]
        C = df_sim[(df_sim.Tabla == "Absoluta")&(df_sim["Posición"] == 16)]["Equipo"].iloc[0]
        # Ponderada
        X = df_sim[(df_sim.Tabla == "Ponderada")&(df_sim["Posición"] == 18)]["Equipo"].iloc[0]
        Y = df_sim[(df_sim.Tabla == "Ponderada")&(df_sim["Posición"] == 17)]["Equipo"].iloc[0]
        Z = df_sim[(df_sim.Tabla == "Ponderada")&(df_sim["Posición"] == 16)]["Equipo"].iloc[0]

        if A == X: 
            l1a += 1
            if B == Y: 
                l2a += 1
                relegation_matches.append([n_sim, C, Z])
            elif B != Y: 
                l2b += 1
                if B == Z: l3a +=1
                elif B != Z: 
                    l3b += 1
                    relegation_matches.append([n_sim, B, Z])
        elif A != X: 
            l1b += 1
            if A == Y: 
                l2c += 1
                if B == X: 
                    l3c += 1
                    relegation_matches.append([n_sim, C, Z])
                elif B != X: 
                    l3d += 1
                    relegation_matches.append([n_sim, B, Z])
            elif A != Y: 
                l2d += 1
                if B == X:
                    l3e += 1
                    matches_cd3.append([n_sim, Y, C, A, X, B])
                    relegation_matches.append([n_sim, Y, Z])
                elif B != X: 
                    l3f += 1
                    relegation_matches.append([n_sim, B, Y])

    print("A = X", l1a/N_sim)
    print("A != X", l1b/N_sim)
    print("-------------------")
    print("A = X & B = Y", l2a/N_sim)
    print("A = X & B != Y", l2b/N_sim)
    print("A != X & A = Y", l2c/N_sim)
    print("A != X & A != Y", l2d/N_sim)
    print("-------------------")
    print("A = X & B != Y & B = Z", l3a/N_sim)
    print("A = X & B != Y & B != Z", l3b/N_sim)
    print("A != X & A = Y & B = X", l3c/N_sim)
    print("A != X & A = Y & B != X", l3d/N_sim)
    print("A != X & A != Y & B = X", l3e/N_sim)
    print("A != X & A != Y & B != X", l3f/N_sim)
    df_cd3 = pd.DataFrame(matches_cd3, columns = ["n_sim","PenPon","AntAbs","Des1","Des2","PenAbs"])
    df_rel_matches = pd.DataFrame(relegation_matches, columns = ["n_sim","team_1","team_2"])
    df_cd3.to_csv("df_cd3.csv")
    df_rel_matches.to_csv("df_rel_matches.csv")
    return (df_cd3, df_rel_matches)

def create_summary_reasons(df_desc_stats, N_sim):
    prob_desc = df_desc_stats.Equipo.value_counts()/N_sim
    summary_reasons = pd.pivot_table(df_desc_stats, index = "Equipo", columns = "Desc", values = "Motivo", 
                                     aggfunc = "count").fillna(0)/N_sim
    summary_reasons = summary_reasons.loc[prob_desc.index]
    summary_reasons.columns = [1,2,3]
    summary_reasons["Prob Desc"] = summary_reasons[1] + summary_reasons[2] + summary_reasons[3]
    summary_reasons.to_excel("resumen_descenso.xlsx")
    return summary_reasons

def fit_poisson_model(df):
    df_played = df[df.Jugado == 1]
    goal_model_data = pd.concat([df_played[['Local','Visita','GL']].assign(Localia=1).rename(
                columns={'Local':'Equipo','Visita':'Rival','GL':'Goles'}),
               df_played[['Visita','Local','GV']].assign(Localia=0).rename(
                columns={'Visita':'Equipo','Local':'Rival','GV':'Goles'})])

    poisson_model = smf.glm(formula="Goles ~ Localia + Equipo + Rival", data = goal_model_data, 
                            family=sm.families.Poisson()).fit()
    #poisson_model.summary()
    return poisson_model

def plot_poisson_dist(df):
    N_poisson = 8
    poisson_pred = np.column_stack([[poisson.pmf(i, df[j].mean()) for i in range(N_poisson)] for j in ['GL','GV']])

    #Graficar distribución actual de goles en el torneo
    plt.hist(df[['GL','GV']].values, range(9), density = True, label = ['Local','Visita'], 
             alpha = 0.8, color=["#FFA07A", "#20B2AA"])

    #Graficar predicción de goles según tasas de poisson (promedios local-visita)
    pois1, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,0], 
                      linestyle ='-', marker = 'o', label ='Local', color ='#CD5C5C', ms = 15)
    pois2, = plt.plot([i-0.5 for i in range(1,9)], poisson_pred[:,1], 
                      linestyle= '-', marker= 'o', label ='Visita', color ='#006400', ms = 15)

    #Atributos del gráfico
    leg = plt.legend(loc ='upper right', fontsize = 20, ncol = 2)
    leg.set_title("Poisson         Actual", prop = {'size':'15', 'weight':'bold'})
    plt.xticks([i-0.5 for i in range(1,9)],[i for i in range(9)], size = 20)
    plt.yticks(size = 12)
    plt.xlabel("Goles por partido",size=20)
    plt.ylabel("Proporción de partidos (%)",size=20)
    plt.title("Número de goles por partido (Primera División Chile 2020)",size=20,fontweight='bold')
    plt.ylim([-0.004, 0.4])
    plt.tight_layout()
    plt.savefig("poisson_dist.png")
    plt.show()
    
def sim_poisson_modification(team, N_matches, goles_team, goles_rival,sim_poisson_local, sim_poisson_visita):

    local_matches = sim_poisson_local[sim_poisson_local.index.get_level_values(1) == team]
    rival_local_matches = sim_poisson_visita[sim_poisson_visita.index.get_level_values(0).isin(local_matches.index.get_level_values(0))]
    visita_matches = sim_poisson_visita[sim_poisson_visita.index.get_level_values(1) == team]
    rival_visita_matches = sim_poisson_local[sim_poisson_local.index.get_level_values(0).isin(visita_matches.index.get_level_values(0))]

    sim_poisson_local_mod = sim_poisson_local.copy()
    sim_poisson_visita_mod = sim_poisson_visita.copy()

    #cambiar sim_poisson_local y sim_poisson_visita
    for id_match in visita_matches.head(N_matches).index.get_level_values(0):
        sim_poisson_visita_mod.loc[id_match, team] = goles_team
        sim_poisson_local_mod.loc[id_match] = goles_rival
        print(id_match)


    for id_match in local_matches.head(N_matches - visita_matches.shape[0]).index.get_level_values(0):
        sim_poisson_local_mod.loc[id_match, team] = goles_team
        sim_poisson_visita_mod.loc[id_match] = goles_rival
        print(id_match)
        
    return sim_poisson_local_mod, sim_poisson_visita_mod

def probs_relegation(team, df_posicion, N_sim):
    df_ult_abs = df_posicion[(df_posicion.Tabla == "Absoluta")&
            (df_posicion["Posición"] == 18)&
            (df_posicion["Equipo"] == team)]
    df_pen_abs = df_posicion[(df_posicion.Tabla == "Absoluta")&
            (df_posicion["Posición"] == 17)&
            (df_posicion["Equipo"] == team)]
    df_ant_abs = df_posicion[(df_posicion.Tabla == "Absoluta")&
            (df_posicion["Posición"] == 16)&
            (df_posicion["Equipo"] == team)]
    
    df_ult_pon = df_posicion[(df_posicion.Tabla == "Ponderada")&
            (df_posicion["Posición"] == 18)&
            (df_posicion["Equipo"] == team)]
    df_pen_pon = df_posicion[(df_posicion.Tabla == "Ponderada")&
            (df_posicion["Posición"] == 17)&
            (df_posicion["Equipo"] == team)]
    
    df_ant_pon = df_posicion[(df_posicion.Tabla == "Ponderada")&
            (df_posicion["Posición"] == 16)&
            (df_posicion["Equipo"] == team)]
    
    p_ult_abs = df_ult_abs.shape[0]/N_sim
    p_pen_abs = df_pen_abs.shape[0]/N_sim
    p_ant_abs = df_ant_abs.shape[0]/N_sim
    
    p_ult_pon = df_ult_pon.shape[0]/N_sim
    p_pen_pon = df_pen_pon.shape[0]/N_sim
    p_ant_pon = df_ant_pon.shape[0]/N_sim
    
    print("Probs Abs", p_ult_abs, p_pen_abs, p_ant_abs)
    print("Probs Pon", p_ult_pon, p_pen_pon, p_ant_pon)
    
    p_pos_pon = df_posicion[(df_posicion.Tabla == "Ponderada") &
            (df_posicion.Equipo == team)]["Posición"].value_counts().sort_index(ascending = False)*100/N_sim

    bar_color = []
    legends = []
    for value in p_pos_pon.index:
        if value == 18:
            bar_color.append("salmon")
            legends.append(["Descenso Directo"])
        elif value in [16,17]:
            bar_color.append("orange")
            legends.append(["Posible Liguilla"])
        else:
            bar_color.append("green")
            legends.append(["Salvado"])

    plt.bar(p_pos_pon.index, p_pos_pon.values, color = bar_color)

    for index in p_pos_pon.index:
        value = p_pos_pon[index]
        if (value > 1) & (index != 18) & (index >= 10):
            plt.text(index - 0.4, value + 0.5, fontsize = 20, s = str(int(round(value)))+"%")
        if index == 18:
            plt.text(index - 0.4, value + 0.5, fontsize = 20, s = str((round(value,1)))+"%")

    custom_lines = [Line2D([0], [0], color="r", lw=4),
                    Line2D([0], [0], color="orange", lw=4),
                    Line2D([0], [0], color="green", lw=4)]

    plt.legend(custom_lines, ['Descenso Directo', 'Posible Liguilla', 'No Desciende'], fontsize = 25)

    plt.xlabel("Posición", fontsize = 25)
    plt.ylabel("% en Posición", fontsize = 25)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.title("Posición Final "+team+"- Tabla Ponderada", fontsize = 30)
    plt.xlim(10,19)
    plt.ylim(0,max(p_pos_pon)*1.1)
    plt.show()