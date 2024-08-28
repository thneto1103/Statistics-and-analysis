import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

pibLen = 265
expecLen = 238

def extract_data(fileName):
    df = pd.read_csv(fileName)
    return df

def takeAwayQualiIDH(df):
    df = df.drop(columns = ['qualiIDH'])
    return df

def takeAwayCodes(df):
    df = df.drop(columns = ['Code'])
    return df

def treatDf(df):
    auxdf = takeAwayCodes(df)
    finaldf = takeAwayQualiIDH(auxdf)
    return finaldf

def merge(df1,df2):
    return df1.merge(df2,how = 'left')

def newMain():
    pibPerCapitadf = extract_data('./PibPerCapitaFormated.csv')     #Extrai o dataFrame do arquivo .csv
    expecVidadf = extract_data('ExpecVidaFormated.csv')
    IDHdf = extract_data('IDH.csv')
    df = pibPerCapitadf.merge(expecVidadf, left_on = 'Code' , right_on = 'Code', how = 'left')  #junta as informações de Pib e expectativa no mesmo dataFrame
    df = df.dropna(subset = ["LifeExpec"])                      #Trata valores vazios nas colunas 
    df = df.dropna(subset = ["PibPerCapita"])
    df.replace(',','', regex=True, inplace=True)
    df['LifeExpec'] = df['LifeExpec'].astype(float)      #Tratando as Colunas para float para poder efetuar media e mediana
    df['PibPerCapita'] = df['PibPerCapita'].astype(float)
    LifeExpecMean = df['LifeExpec'].mean()
    PibPerCapitaMean = df['PibPerCapita'].mean()
    PibPerCapitaMedian = df['PibPerCapita'].median()
    LifeExpecMedian = df['LifeExpec'].median()
    LifeExpecStdd = df['LifeExpec'].std()
    PibPerCapitaStdd = df['LifeExpec'].std()
    new_index = range(0,240)
    df = df.reset_index(drop = True)
    binsPibPerCapita = [0,1000,5000,10000,50000,200000]                  #Tabela de frequencia do Pib per Capita
    labelsPibPerCapita = ['< 1000', '1000 - 4999', '5000 - 9999', '10000 - 49999' , '> 50000' ]
    FreqTablePibPerCapitaS = pd.Series()
    FreqTablePibPerCapitaS = pd.cut(df['PibPerCapita'], bins = binsPibPerCapita , labels = labelsPibPerCapita , right =False)
    frequenciaPibPerCapita = FreqTablePibPerCapitaS.value_counts().sort_index()
    binsLifeExpec = [0,60,70,80,90]                              #Tabela de frequencia da expectativa de vida
    labelsLifeExpec = ['< 60 ', '60 - 69', '70 - 79' , '> 80' ]
    FreqTableLifeExpecS = pd.Series()                                  
    FreqTableLifeExpecS = pd.cut(df['LifeExpec'], bins = binsLifeExpec , labels = labelsLifeExpec , right =False)
    frequenciaLifeExpec = FreqTableLifeExpecS.value_counts().sort_index()
    IDHdf = IDHdf.rename(columns = {'ISO3':'Code'})                                 #Trata a matriz dos IDHs
    IDHdf = IDHdf.rename(columns = {'Human Development Groups':'qualiIDH'})
    IDHdf = IDHdf.rename(columns = {'HDI Rank (2021)':'quantIDH'})
    labelsIDH = ['Low','Medium','High','Very High']                                 #Tabela de frequencia dos IDHs qualitativos
    frequenciaIDH = IDHdf['qualiIDH'].value_counts()
    dfComplete = df.merge(IDHdf, left_on = 'Code' , right_on = 'Code', how = 'left')            #Produz a matriz com o valor dos IDHs adicionados
    dfComplete = dfComplete.dropna(subset = ["qualiIDH"])
    dfComplete = dfComplete.dropna(subset = ["quantIDH"])
    dfComplete = dfComplete.reset_index(drop = True)
    floatDfComplete = treatDf(dfComplete)                   #Retira todos os valores string das colunas
    floatDf = takeAwayCodes(df)
    EducatedPopulationDf = pd.read_csv("FormalEducatedPopulationF.csv")        #Tratando indice de educação da população
    EducatedPopulationDf = EducatedPopulationDf.dropna(subset = ["Code"])
    EducatedPopulationDf = EducatedPopulationDf.dropna(subset = ["FormalEducatedPopulation"])
    EducatedPopulationDf = EducatedPopulationDf.reset_index(drop = True)
    dfComplete = dfComplete.merge(EducatedPopulationDf, left_on = 'Code' , right_on = 'Code', how = 'left') 
    dfComplete = dfComplete.dropna(subset = ["FormalEducatedPopulation"])
    dfComplete = dfComplete.reset_index(drop = True)
    binsEducatedPopulation = [0,90,95,100]                              #Tabela de frequencia da educação da populaçao
    labelsEducatedPopulation = ['Pouca Educação Formal', 'Educação Formal Mediana', 'Alta Educação Formal' ]
    FreqTableEducatedPopulationS = pd.Series()                                  
    FreqTableEducatedPopulationS = pd.cut(dfComplete['FormalEducatedPopulation'].astype(float), bins = binsEducatedPopulation , labels = labelsEducatedPopulation , right =False)
    frequenciaEducatedPopulationS = FreqTableEducatedPopulationS.value_counts().sort_index()
    print(floatDfComplete)


    #ax1 = plt.subplot()                                     #Plota o gráfico de pizza do idh qualitativo
    #ax1.pie(frequenciaIDH, labels = labelsIDH, autopct = '%1.1d%%' , startangle = 90 )
    #ax1.axis('equal')
    #plt.show()


    #plot = df.plot(x = 'Code', kind = "bar", color = "red")                #Plota grafico de barras de todos os dados de df vs Code
    #dfOrdered = dfComplete.sort_values(by = 'PibPerCapita')                                #Plota o mesmo gráfico acima porém ordenado
    #dfOrdered = dfOrdered.sort_values(by = 'LifeExpec')
    #dfOrdered = dfOrdered.sort_values(by = 'quantIDH')
    #orderedPlot = dfOrdered.plot(x = 'Code', kind = "bar")
    #plot.set_xticklabels([])
    #orderedPlot.set_xticklabels([])
    #plt.show()


    normalizedFloatDfComplete = floatDfComplete.copy()                      #Cria um dataFrame normalizado a partir do dataframe completo
    normalizedFloatDfComplete['PibPerCapita'] = normalizedFloatDfComplete['PibPerCapita'].apply(lambda x: x/floatDfComplete.max().iloc[0])
    normalizedFloatDfComplete['LifeExpec'] = normalizedFloatDfComplete['LifeExpec'].apply(lambda x: x/floatDfComplete.max().iloc[1])
    normalizedFloatDfComplete['quantIDH'] = normalizedFloatDfComplete['quantIDH'].apply(lambda x: x/floatDfComplete.max().iloc[2])
    #normalizedFloatDfComplete.plot(y = 'PibPerCapita')
    #normalizedFloatDfComplete.plot(y = 'LifeExpec')
    #normalizedFloatDfComplete.plot(y = 'quantIDH')                               #printa as colunas do dataFrame vs o index de cada um
    #normalizedFloatDfComplete.plot.scatter(y = 'quantIDH', x = 'PibPerCapita')
    #normalizedFloatDfComplete.plot.scatter(y = 'LifeExpec', x = 'PibPerCapita', color = 'red')
    #plt.show()


    ########Cria painel 3x3 com histograma , correlação e Scatter ##############
    # Criar uma figura 3x3
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Lista de colunas
    columns = normalizedFloatDfComplete.columns

    # Plotar os gráficos
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            ax = axes[i, j]
        
            if i == j:
                # Diagonal: Histograma
                sns.histplot(normalizedFloatDfComplete[col1], bins=20, kde=True, ax=ax)
                #ax.set_title(f'Histograma de {col1}')
            elif i < j:
                # Acima da diagonal: Scatter plot com linha de regressão polinomial
                sns.scatterplot(x=normalizedFloatDfComplete[col1], y=normalizedFloatDfComplete[col2], ax=ax)
                z = np.polyfit(normalizedFloatDfComplete[col1], normalizedFloatDfComplete[col2], 1)
                p = np.poly1d(z)
                ax.plot(normalizedFloatDfComplete[col1], p(normalizedFloatDfComplete[col1]), color='red')
                #ax.set_title(f'{col1} vs {col2}')
            else:
                # Abaixo da diagonal: Correlação
                corr = normalizedFloatDfComplete[col1].corr(normalizedFloatDfComplete[col2])
                ax.text(0.5, 0.5, f'{corr:.2f}', horizontalalignment='center', verticalalignment='center', fontsize=10)
                ax.set_axis_off()
        
            if j == 0:
                ax.set_ylabel(col1)
            if i == len(columns) - 1:
                ax.set_xlabel(col2)

    # Ajustar o layout
    plt.tight_layout()
    plt.show()

    #######ScatterPlot PibPerCapita vs quantIDH#############
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PibPerCapita', y='quantIDH', data=normalizedFloatDfComplete, label='Data Points')

    # Polynomial fit (degree 1, i.e., linear regression)
    z = np.polyfit(normalizedFloatDfComplete['PibPerCapita'], normalizedFloatDfComplete['quantIDH'], 1)  # For a linear fit, change 1 to the desired degree
    p = np.poly1d(z)

    # Plotting the polynomial regression line
    x_vals = np.linspace(normalizedFloatDfComplete['PibPerCapita'].min(), normalizedFloatDfComplete['PibPerCapita'].max(), 100)
    plt.plot(x_vals, p(x_vals), color='red', label=f'Polynomial Fit (degree 1)')

    # Customizing the plot
    plt.title('Scatter Plot with Polynomial Regression Line')
    plt.xlabel('PibPerCapita')
    plt.ylabel('quantIDH')
    plt.legend()
    plt.show()

    #######ScatterPlot PibPerCapita vs LifeExpec#############
    plt.figure(figsize=(10, 6))
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PibPerCapita', y='LifeExpec', data=normalizedFloatDfComplete, label='Data Points')

    # Polynomial fit (degree 1, i.e., linear regression)
    z = np.polyfit(normalizedFloatDfComplete['PibPerCapita'], normalizedFloatDfComplete['LifeExpec'], 1)  # For a linear fit, change 1 to the desired degree
    p = np.poly1d(z)

    # Plotting the polynomial regression line
    x_vals = np.linspace(normalizedFloatDfComplete['PibPerCapita'].min(), normalizedFloatDfComplete['PibPerCapita'].max(), 100)
    plt.plot(x_vals, p(x_vals), color='red', label=f'Polynomial Fit (degree 1)')

    # Customizing the plot
    plt.title('Scatter Plot with Polynomial Regression Line')
    plt.xlabel('PibPerCapita')
    plt.ylabel('LifeExpec')
    plt.legend()
    plt.show()


    #######ScatterPlot LifeExpec vs quantIDH#############
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='quantIDH', y='LifeExpec', data=normalizedFloatDfComplete, label='Data Points')
        # Polynomial fit (degree 1, i.e., linear regression)
    z = np.polyfit(normalizedFloatDfComplete['quantIDH'], normalizedFloatDfComplete['LifeExpec'], 1)  # For a linear fit, change 1 to the desired degree
    p = np.poly1d(z)

    # Plotting the polynomial regression line
    x_vals = np.linspace(normalizedFloatDfComplete['quantIDH'].min(), normalizedFloatDfComplete['quantIDH'].max(), 100)
    plt.plot(x_vals, p(x_vals), color='red', label=f'Polynomial Fit (degree 1)')

    # Customizing the plot
    plt.title('Scatter Plot with Polynomial Regression Line')
    plt.xlabel('quantIDH')
    plt.ylabel('LifeExpec')
    plt.legend()
    plt.show()


    ax1 = plt.subplot()                                     #Plota o gráfico de pizza do indice de educação qualitativo
    ax1.pie(frequenciaEducatedPopulationS, labels = labelsEducatedPopulation, autopct = '%1.1d%%' , startangle = 90 )
    ax1.axis('equal')
    plt.show()


    print(floatDf.cov())                          #Printa a matriz de covariancia e correlação
    print("\n")
    print(floatDfComplete.cov())
    print(floatDf.corr())
    print("\n")
    print(floatDfComplete.corr())
    print(LifeExpecStdd)
    
newMain()