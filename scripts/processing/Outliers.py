def outliers(datas, Q1, Q3, column):
    if (column not in datas.columns):
        print('ERRO/outliers - procura_outiliers: column não existe no dataframe')
        return 0 
    try: 
        Q1 = float(Q1)
        Q3 = float(Q3)
        column = str(column)
    except ValueError:
        print('ERRO/outliers - procura_outiliers: parâmetros não estão no formato correto (e não podem ser convertidos)')
        return 0
    Q1 = datas[column].quantile(Q1)
    Q3 = datas[column].quantile(Q3)
    IQR = Q3 - Q1
    datasOut = datas.loc[((datas[column] > (Q1 - 1.5 * IQR)) & (datas[column] < (Q3 + 1.5 * IQR)))]
    outliers = datas.loc[((datas[column] < (Q1 - 1.5 * IQR)) | (datas[column] > (Q3 + 1.5 * IQR)))]
    return datasOut, outliers
