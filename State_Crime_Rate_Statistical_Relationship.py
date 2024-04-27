import pandas
import numpy
import inspect
import functools

from numpy import inf

from math import sqrt
from math import pi
from math import factorial
from math import isnan

from itertools import chain

import matplotlib.pyplot as plt

from scipy.stats import norm as normal_probability
from scipy.stats import t as t_probability
from scipy.stats import probplot
from scipy.stats import chisquare as chi_square
from scipy.stats import chi2_contingency
def chi_square_contingency(data): return chi2_contingency(data, correction = False)
def chi_2_expected(data): return chi_square_contingency(data)[3]
from scipy.stats import chi2 as chi_square_probability
from scipy.stats import f as f_probability
from scipy.stats import linregress

from sklearn import linear_model

### Display and rounding code and functions:

pandas.options.display.float_format = '{:.4f}'.format
numpy.set_printoptions(precision=4)
pandas.set_option('display.max_rows', 25)
pandas.set_option('display.max_columns', 10)

def truncate(func):
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        result = func(*args, **kwargs)
        #print(result, type(result))
        if isinstance(result, (numpy.ndarray)): result = pandas.Series(result)
        if isinstance(result, (list, tuple, set, pandas.Series)):
            orig_type = type(result)   
            truncated = [float(x) for x in result]
            #print(type(truncated), truncated)
            for i in range(int(len(truncated))):
                #print(i, type(truncated[i]), truncated[i])
                if type(truncated[i]) == numpy.float64: truncated[i] = float(truncated[i])
                if truncated[i] != inf and truncated[i] != -inf and truncated[i] != 0 and not isnan(truncated[i]): 
                    #print(".")
                    truncated[i] = int((truncated[i] + abs(truncated[i]) / truncated[i] * 0.0000500001) * 10000) / 10000.0
            result = orig_type(truncated)
        else:
            #print("*")
            if type(result) == numpy.float64: result = float(result)
            if result != inf and result != -inf and result != 0 and not isnan(result):
                result = int((result + abs(result) / result * 0.0000500001) * 10000) / 10000.0
        #print("x")
        return result
    return wrap_func

### Helpful functions:

def combination(n, r): 
    n = int(n + 0.00001)
    r = int(r + 0.00001)
    return factorial(n) / (factorial(r) * factorial(n - r))
def permutation(n, r): 
    n = int(n + 0.00001)
    r = int(r + 0.00001)
    return factorial(n) / factorial(n - r)


### Some graphics things:

plt.style.use('ggplot')

### this is a visual trick for probability distributions
### P(X) returns the probability values for the whole pd; P(x) returns the probability for just P(x)
### may be nice to set things up so that P(x1, x2, x3, ...) works, but saving that for later
### right now, won't do anything if an iterable is passed that isn't the whole index
def P(value):
    try:
        iter(value)
        if list(value) == list(data.index): return data[data.columns[0]]
    except TypeError:
        if value in data.index: return data.loc[value, data.columns[0]]
        else: return 0
        
## aid for creating binomial probability distribution with our data
def binomial_probability_distribution(success_probability, trials):
    values = []
    for successes in range(0, trials + 1):
        values.append(probability(binomial, success_probability=success_probability, trials=trials, successes=successes))
    return values             
    
### Python training wheels:

def please_define(*variables):
    for variable in variables:
        if type(variable) == None:
            print("Something needed to be initialized!  See help for the function you called.")
            return False
    return True
    
def load(option, *args, sample = False):
    """
    #Usage:
    load(data_set) or load(manually)
    
    #Effect: data is loaded in variable named: data
    #See help(data_set) or help(manually) if needed
    """
    global population
    global data
    global data_bk
    global table
    global probability_distribution
    #global P
    global X
    
    population = False
    option(*args)
    if sample:
        data = data.sample(sample)
    data_bk = data.copy()
    table = data.copy()
    if probability_distribution:
        #P = data[data.columns[0]]
        X = data.index
        probability_distribution = False
    else:
        #P = None
        X = None
    return data

def reload():
    global data
    global data_bk
    global table
    data = data_bk.copy()
    table = data.copy()
    return data

def filter(data_filtered):
    global data
    global table
    data = data[data_filtered]
    table = data.copy()
    return data

def data_set():
    """
    #Example usage:
    chapter = "10"
    set_name = "carprice"
    load(data_set)
    
    #Effect: data is loaded in variable named: data
    """
    global data
    global chapter
    global set_name
    global gss_historical
    global gss_current
    
    if set_name.lower() == "gss":
        try:
            data = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss2021.xlsx", engine = 'openpyxl') 
        except ValueError:
            data = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss2021.xlsx")
    elif set_name.lower() == "gss_historical_comparison":
        try:
            gss_historical = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss_2010_to_2018.xlsx", engine = 'openpyxl') 
        except ValueError:
            gss_historical = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss_2010_to_2018.xlsx")
        data = gss_historical
        try:
            gss_current = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss2021.xlsx", engine = 'openpyxl') 
        except ValueError:
            gss_current = pandas.read_excel("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/data_sets/gss_2021/gss2021.xlsx")
    else:
        ## THIS IS EDITED TO BREAK THE OLD LAROSE TEXTBOOK EAY OF DOING THINGS; NO CHAPTER
        data = pandas.read_csv("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/" + set_name + ".csv")
        """
        #LAROSE:
        if type(chapter) == int: chapter = str(chapter)
        if len(chapter) == 1: chapter = "0" + chapter
        data = pandas.read_csv("https://gannon.blackboard.com/bbcswebdav/users/nogaj001/math213/chapter_" + chapter + "/" + set_name + ".csv")
        """
    
    chapter = None
    set_name = None
    return data

def manually():
    """
    #Example usage:
    values = (3, 4, 5, 6), (5, 6, 7, 8)
    labels = "x", "y" #this step is optional
    indexes = "a", "b", "c", "d" #this step is optional
    load(data_set)
    
    #Effect: data is loaded in variable named: data
    """
    global data
    global values
    global labels
    global indexes
    global index_label
    
    if labels:
        if type(labels) != tuple and type(labels) != list: labels = (labels,)
        if len(labels) != len(values): values = (values,)
    if labels: data = pandas.DataFrame(dict(zip(labels, values)), index=indexes)
    else: data = pandas.DataFrame(numpy.array(values).T, index=indexes)
    
    if index_label: data.index.name = index_label
    
    values = None
    labels = None
    indexes = None
    index_label = None
    return data

def add_categorical_factor(variable, filter_dict, name = "FACTOR", separate = False):
    global data
    if separate: new_data = pandas.DataFrame({label:data.loc[option][variable] for label, option in filter_dict.items()})#.T.reset_index().drop("index", axis=1)
    else:
        new_data = pandas.DataFrame({name:[""] * len(data)})
        for label, option in filter_dict.items(): new_data.loc[option, name] = label
    data = pandas.concat([new_data, data], axis=1)
    return data
    
def keep(*columns):
    global data
    data = data[list(columns)]
    return data

def eliminate(*columns):
    global data
    data.drop(list(columns), axis = 1, inplace = True)
    return data
    
    

def make(option, *args):
    """
    #Usage:
    make(frequency_distribution),
    make(copy_data),
    make(crosstabulation)
    
    #Effect: a table is created in variable named: table
    #See help(frequency_distribution), help(copy_data), help(crosstabulation) if needed
    """
    global data
    global group_by
    if group_by == None:
        group_by = data.columns[0]
    return option(*args)

def frequency_distribution():
    """
    #Example usage:
    chapter = "02"
    set_name = "worldwater"
    load(data_set)
    group_by = "Continent"
    make(frequency_distribution)
    
    #Effect: frequency distribution is loaded in variable named: table
    """
    global data
    global table
    global group_by
    global bins
    global frequency
    global relative
    global cumulative
    global cumulative_relative
    global append
    global index_sort
    global show_nan
    global hist_relative
    global hist_bins
    global hist_group_by

    hist_relative = relative
    hist_bins = bins
    hist_group_by = group_by
    
    if not please_define(data, group_by): return
    
    if not append:
        if bins == None: # assume categories
            table = data[group_by].value_counts(sort=False, dropna=not show_nan).to_frame()
            if index_sort:
                table.sort_index(key=lambda x: x.str.lower() if (type(x) == str) else x , inplace=True)

        else: # assume quantitative column data
            table = pandas.cut(data[group_by], bins, include_lowest=True, right=False).value_counts(dropna=not show_nan).to_frame()
            table = table.reindex(sorted(table.index.categories))
        table.columns = [str(group_by) + " Frequency"]
    
    table[str(group_by) + " Relative Frequency"] = table[str(group_by) + (" Frequency" if not append else "")] / len(data.index)
    if cumulative: table[str(group_by) + " Cumulative Frequency"] = table[str(group_by) + (" Frequency" if not append else "")].cumsum()
    if cumulative_relative: table[str(group_by) + " Cumulative Relative Frequency"] = table[str(group_by) + " Relative Frequency"].cumsum()
    if not frequency: table = table.drop(str(group_by) + (" Frequency" if not append else ""), axis=1)
    if not relative: table = table.drop(str(group_by) + " Relative Frequency", axis=1)
    
    
    group_by = None
    bins = None
    append = False
    relative = False
    cumulative = False
    cumulative_relative = False
    frequency = True
    index_sort = False
    show_nan = False
    return table

def crosstabulation():
    """
    #Example usage:
    chapter = "02"
    set_name = "worldwater"
    load(data_set)
    group_by = "Continent"
    include = "Climate", "Main use"
    hide_empty = False # try this both ways
    make(crosstabulation)
    
    #Effect: crosstabulation is loaded in variable named: table
    """
    global data
    global table
    global group_by
    global include
    global hide_empty
    global show_totals
    
    if not please_define(data, group_by, include): return
    if type(include) != tuple: include = (include,)
    
    other_columns = []
    for column in include: other_columns.append(data[column])

    table = pandas.crosstab(data[group_by], other_columns, dropna=hide_empty, margins=show_totals, margins_name="TOTAL")
    
    group_by = None
    include = None
    hide_empty = True
    show_totals = False
    return table

def copy_data():
    """
    #Example usage:
    values = (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), ("A","B","C","D","E") 
    labels = "x", "y", "k"  
    indexes = "a", "b", "c", "d", "e" 
    load(manually)
    make(copy_data)
    
    #Effect: data has been copied directly into a variable named: table
    """
    global data
    global table
    
    table = data.copy()

    return table

def show(option, *args, **kwargs):
    """
    #Usage:
    show(bar_graph),
    show(histogram),
    show(dotplot),
    show(scatter),
    show(five_number_summary)
    show(describe),
    show(boxplot)
    
    #Effect: graph or summary is displayed, and created in variable named: graph
    #See help(bar_graph), help(histogram), help(dotplot), help(scatter), help(boxplot) if needed
    """
    global data
    global group_by
    global column
    if group_by == None:
        group_by = data.columns[0]
    return option(*args, **kwargs)

def bar_graph():
    """
    #Example usage:
    values = (1, 2, 3, 4, 5), (6, 7, 8, 9, 10), ("A","B","C","D","E") 
    labels = "x", "y", "k"  
    indexes = "a", "b", "c", "d", "e" 
    load(manually)
    make(copy_data)
    show(bar_graph)
    
    #Effect: a bar graph is displayed with all quantitative variables of variable table, and created in variable named: graph
    """
    
    global table
    global graph
    total_column = False
    total_row = False
    if table.columns[-1] == "TOTAL":
        last_column = table["TOTAL"]
        total_column = True
        table.drop("TOTAL", axis=1, inplace=True)
    if list(table.index)[-1] == "TOTAL":
        last_row = table.tail(1)
        total_row = True
        table.drop(table.tail(1).index, axis=0, inplace=True)
    #graph = table.sort_values(by=list(table.columns)).plot.bar(rot=0)
    #graph = table.sort_index().plot.bar(rot=0)
    graph = table.plot.bar(rot=0)
    if total_row: table.loc["TOTAL"] = last_row.iloc[0]
    if total_column: table["TOTAL"] = last_column
    
    return graph

def histogram():
    """
    #Example usage:
    values = 0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 9, 10
    load(manually)
    group_by = 0
    
    show(histogram)
    
    #Effect: a histogram of column group_by is displayed from data, and created in variable named: graph
    """
    
    global data
    global hist_group_by
    global graph
    global hist_bins
    global hist_relative
   
    graph = data[hist_group_by].hist(bins=hist_bins, grid=False, edgecolor="black", density=hist_relative)
    
    return graph

def dotplot():
    """
    #Example usage:
    values = 0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 9, 10
    load(manually)
    group_by = 0
    
    show(dotplot)
    
    #Effect: a dotplot of column group_by is displayed from data, and created in variable named: graph
    """
    
    global data
    global group_by
    global graph
   
    x = data.copy()
    #x = x.sort_values(group_by, axis=1)
    x = x.sort_values(group_by)
    x = x.reset_index(drop=True)
    x["_"] = 1
    for i in range(1,len(x.index)):
        if x.at[i, group_by] == x.at[i - 1, group_by]: x.at[i, "_"] = x.at[i - 1, "_"] + 1
    #plot = x.plot.scatter(group_by, "_", figsize=(6.25, 1.0/6.0*max(x["_"])))
    #graph = plot.show()
    graph = x.plot.scatter(group_by, "_", figsize=(6.25, 1.0/6.0*max(x["_"])), yticks = [1, x["_"].max()])
    group_by = None

    return graph

def scatter(x_col = None, y_col = None, connect_lines = None):
    """
    #Example usage:
    chapter = "2"
    set_name = "murderrate"
    load(data_set)
    x_column = "Year"
    y_column = "Rate"
    lines = True
    show(scatter)
    
    #Effect: a scatterplot of columns x_column, y_column is displayed from data, and created in variable named: graph
    """
    
    global data
    global lines
    global x_column
    global y_column
    global graph
   
    if x_col == None: x_col = x_column
    if y_col == None: y_col = y_column
    if connect_lines == None: connect_lines = lines
    if connect_lines: graph = data.plot.line(x_col, y_col)
    else: graph = data.plot.scatter(x_col, y_col)
    
    lines = False
    x_column = None
    y_column = None

    return graph

def boxplot():
    """
    #Example usage:
    chapter = "03"
    set_name = "dietarysupp"
    load(data_set)
    group_by = "Usage (in millions) "
    
    show(boxplot)
    
    #Effect: a boxplot of column group_by is displayed from data, and created in variable named: graph
    """
    
    global data
    global group_by
    global graph
   
    x = data[group_by].copy().to_frame()
    graph = x.boxplot(column=group_by, grid=False, vert=False)
    
    group_by = None

    return graph

def full_summary():
    """
    #Example usage:
    values = 0, 1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7, 9, 10
    labels = "x"
    load(manually)
    show(full_summary)
    
    #Effect: a full summary chart is displayed from data, and created in variable graph
    """
    
    global data
    global graph
    
    graph = data.describe()
    
    return graph 

def five_number_summary(column = None):
    """
    #Example usage:
    values = 0, 1, 2, 2, 2, 3, 4, 5, 5, 5, 6, 7, 9, 10
    labels = "x"
    load(manually)
    show(five_number_summary, "x")
    
    #Effect: a five number summary chart is displayed from data in column, and created in variable graph
    """
  
    global data
    global graph
    
    if column == None:
        column = data.columns[0]
    graph = data[column].describe().iloc[3:8].to_frame()
    
    return graph   

def normal_probability_plot(show_detail=False):
    """
    #Example usage:
    values = 0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 9, 10
    load(manually)
    group_by = 0
    
    show(dotplot)
    
    #Effect: a dotplot of column group_by is displayed from data, and created in variable named: graph
    """
    
    global data
    global group_by
    global graph
   
    graph = probplot(data[group_by].dropna(), dist="norm", plot=plt)
    
    if show_detail: return graph

def perform(option, *args, **kwargs):
    """
    #Usage:
    show(anova)
    
    #Effect: corresponding summary is displayed, and various results are returned
    #See help(anova), if needed
    """
    global data
    global group_by
    global column
    if group_by == None:
        group_by = data.columns[0]
    return option(*args, **kwargs)

def anova(*args, show = False):
    global data
    k = len(args)
    n_t = sum(stat(size, c) for c in args)
    x_double_bar = sum(stat(size, c) * stat(mean, c) for c in args) / n_t
    MSTR = sum(stat(size, c) * (stat(mean, c) - x_double_bar) ** 2 for c in args)/(k - 1)
    MSE = sum((stat(size, c) - 1) * stat(variance, c) for c in args)/(n_t - k)
    F_data = MSTR / MSE
    SSTR = MSTR * (k - 1)
    SSE = MSE * (n_t - k)
    SST = SSTR + SSE
    df_1 = k - 1
    df_2 = n_t - k
    df = df_1 + df_2
    p_value = 1 - f_probability.cdf(F_data, df_1, df_2)
    anova_table = pandas.DataFrame([\
                  [SSTR,df_1,MSTR,F_data,p_value],\
                  [SSE,df_2,MSE,"",""],\
                  [SST,df,"","",""]],\
                  index = ["Treatment","Error","Total"],\
                  columns = ["Sum of Squares","Degrees of Freedom","Mean Square","F-test Statistic","p-Value"])
    if show:
        print("df_1:", df_1, ", df_2:", df_2)
        print("x_double_bar:", truncate(lambda:x_double_bar)())
        print("SSTR:", truncate(lambda:SSTR)())
        print("SSE:", truncate(lambda:SSE)())
        print("SST:", truncate(lambda:SST)())
        print("MSTR:",truncate(lambda: MSTR)())
        print("MSE:", truncate(lambda:MSE)())
        print("F_data:", truncate(lambda:F_data)())
        print(anova_table)
    return (F_data, p_value, df_1, df_2)

@truncate
def stat(option, *args):
    """
    #Usage:
    #note: *column_name* has to be the actual column name from data that the statistic is being computed on
    stat(mean, *column_name*),
    ...
    
    #Effect: the indicated statistic is computed and used directly; its result is NOT stored in any variable unless manually assigned
    """
    global data
    if len(args) == 0:
        return option(data.columns[0])
    if len(args) == 1: # duct tape for percentile with no column name
        if type(args[0]) == int and args[0] > 0:
            return option(data.columns[0], args[0])
    return option(*args)

def mean(column):
    global data
    return data[column].mean()

def median(column): 
    global data
    return data[column].median()

def mode(column):
    global data
    result = tuple(data[column].mode().tolist())
    if len(result) == 1: return result[0]
    else: return result

def variance(column):
    global data
    global population
    df = 1
    if population: df = 0
    return data[column].var(ddof=df)

def standard_deviation(column):
    global data
    global population
    df = 1
    if population: df = 0
    return data[column].std(ddof=df)

def percentile(column, value):
    global data
    return numpy.nanpercentile(data[column], [value])[0]

#def z_score(column, value):
#    return (value - mean(column)) / standard_deviation(column)

def minimum(column):
    global data
    return data[column].min()

def maximum(column):
    global data
    return data[column].max()

def size(column):
    global data
    return data[column].count()

def number_of_columns(*args):
    global data
    return int(len(data.columns) + 0.00001)


def p_value(option, score_value = None, data_value = None, tails = "two", null_value = 0, scale = None, standard_deviation = 1, alpha = None, graph = False, **kwargs):
    if scale == None: scale = standard_deviation
    if score_value != None:
        data_value = score_value
        null_value = 0
        scale = 1
    
    #kinda duct tape for getting score values from chi2 stuff
    if type(data_value) != float and type(data_value) != int:
        data_value = data_value[0]
    
    data_value = truncate(lambda:data_value)()
    
    if tails == "two":
        location = "ends"
        data_interval = data_value
        if data_interval > null_value: data_interval = (null_value - abs(data_interval - null_value), data_interval)
        else: data_interval = (data_interval, null_value + abs(data_interval - null_value))
    elif tails == "one":
        if data_value < null_value:
            location = "left"
            data_interval = (-inf, data_value)
        else:
            location = "right"
            data_interval = (data_value, inf)
    elif tails == "left":
        location = "left"
        data_interval = (-inf, data_value)
    elif tails == "right":
        location = "right"
        data_interval = (data_value, inf)
        
    cv_scale = {}
    score = truncate(lambda:(data_value - null_value) / scale)()
    if alpha != None:
        if (option == t or option == z) and score_value == None:
            ci = interval(option, probability = 1 - alpha, mean = data_value, standard_deviation = scale, location = "middle", graph = graph, graph_values = [null_value] if location == "ends" else [], **kwargs)

            print((1 - alpha) * 100, "% confidence interval:", ci)
            print(data_value, "+/-", margin_of_error(option, probability = 1 - alpha, mean = data_value, standard_deviation = scale, location = "middle", **kwargs))
            print()
            if location == "ends":
                if ci[0] <= null_value <= ci[1]: print("Confidence interval contains null hypothesis value", null_value, "; Do not reject null hypothesis")
                else: print("Confidence interval does not contain null hypothesis value", null_value, "; Reject null hypothesis")
            else: print("Because this is not a two-tailed test, the confidence interval cannot be used to perform hypothesis test")
            print()
        
        cv = interval(option, probability = alpha, mean = 0, standard_deviation = 1, location = location, graph = False, graph_values = [(data_value - null_value) / scale], **kwargs)
        cv_display = cv
        if cv[0] == -inf: cv_display = cv[1]
        elif cv[1] == inf: cv_display = cv[0]
        
        cv_scale = truncate(lambda:tuple([null_value + scale * v for v in cv]))() 
        if type(cv_display) == float: cv_scale_display = truncate(lambda:null_value + scale * cv_display)() 
        else: cv_scale_display = truncate(lambda:tuple([null_value + scale * v for v in cv_display]))() 
        
        if cv_scale_display != cv_display:
            print("original scale critical values:", cv_scale_display, ", data value:", data_value)
        print("standardized critical values:", cv_display, ", score:", score)
        print()
        
        if cv[0] == -inf or cv[1] == inf:
            if cv[0] <= score <= cv[1]: print("Score", score, "exceeds critical value; Reject null hypothesis")
            else: print("Score", score, "does not exceed critical value; Do not reject null hypothesis") 
        else:
            if cv[0] <= score <= cv[1]: print("Score", score, "does not exceed critical value; Do not reject null hypothesis")
            else: print("Score", score, "exceeds critical value; Reject null hypothesis") 
    else: print("score:", score)
        
    p = probability(option, interval = data_interval, mean = null_value, standard_deviation = scale, location = location, graph = graph, graph_values = cv_scale, **kwargs)
    
    print("p-value:", p)
    
    if alpha != None:
        print()
        if p <= alpha:
            print(p, "<=", alpha, "(p <= alpha); Reject null hypothesis.")
        else:
            print(p, ">", alpha, "(p > alpha); Do not reject null hypothesis.")

    return p

@truncate
def probability(option, *args, **kwargs):
    """
    #Usage:
    #to be completed later
    ...
    
    #Effect: the specific value for the probability distribution indicated is given
    """
    return option(*args, **kwargs, ret = "probability")

@truncate
def interval(option, *args, **kwargs):
    """
    #Usage:
    #to be completed later
    ...
    
    #Effect: the specific value for the probability distribution indicated is given
    """
    return option(*args, **kwargs, ret = "interval")

def alpha_over_2(option, *args, **kwargs):
    """
    #Usage:
    #to be completed later
    ...
    
    #Effect: the specific value for the probability distribution indicated is given
    """
    return interval(option, *args, **kwargs)[1]

@truncate
def margin_of_error(option, *args, **kwargs):
    """
    #Usage:
    #to be completed later
    ...
    
    #Effect: the specific value for the probability distribution indicated is given
    """
    result = interval(option, *args, **kwargs)
    return (result[1] - result[0]) / 2

def binomial(success_probability = None, trials = None, successes = None, ret = "probability"):
    return combination(trials, successes) * success_probability ** successes * (1 - success_probability) ** (trials - successes)
    
def uniform(endpoints = None, interval = None, graph = False, ret = "probability"):
    result = (interval[1] - interval[0]) / (endpoints[1] - endpoints[0]) 

    if graph:
        plt.grid(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        
     
        x = numpy.linspace(endpoints[0], endpoints[1], 1000)
        y = numpy.linspace(1 / (endpoints[1] - endpoints[0]), 1 / (endpoints[1] - endpoints[0]), 1000)
        plt.plot(x, y, 'b')
        
        x_interval = numpy.linspace(interval[0], interval[1], 1000)
        plt.fill_between(x_interval, y, color='r')
        
        plt.plot(numpy.array([endpoints[0] - 1 / 5 * (endpoints[1] - endpoints[0]), endpoints[0]]), numpy.array([0, 0]), color='b')
        plt.plot(numpy.array([endpoints[0], endpoints[0]]), numpy.array([0, 1 / (endpoints[1] - endpoints[0])]), color='b')
        plt.plot(numpy.array([endpoints[1], endpoints[1]]), numpy.array([1 / (endpoints[1] - endpoints[0]), 0]), color='b')
        plt.plot(numpy.array([endpoints[1] + 1 / 5 * (endpoints[1] - endpoints[0]), endpoints[1]]), numpy.array([0, 0]), color='b')
        plt.show()
        
    return result

def z(*args, **kwargs): return normal(*args, **kwargs)
def normal(interval = None, mean = 0, standard_deviation = 1, scale = None, probability = None, location = "left", graph = False, graph_values = [], ret = None):          
    if scale != None: standard_deviation = scale
    if standard_deviation < 0: standard_deviation = -standard_deviation
    if ret == "probability":
        if type(interval) != tuple and type(interval) != list:
            if location == "left": interval = (-inf, interval)
            elif location == "right": interval = (interval, inf)
            else:
                if interval > mean: interval = (mean - abs(interval - mean), interval)
                else: interval = (interval, mean + abs(interval - mean))
        if interval[0] > interval[1]: interval = [interval[1], interval[0]]
        result = normal_probability(mean, standard_deviation).cdf(interval[1]) - normal_probability(mean, standard_deviation).cdf(interval[0])
        if location == "ends": result = 1 - result
        probability = result
        
    #if inspect.getouterframes(inspect.currentframe(), 2)[1][3] == "score":
    if ret == "interval":
        if location == "left": result = (-inf, normal_probability(mean, standard_deviation).ppf(probability))
        if location == "right": result = (normal_probability(mean, standard_deviation).ppf(1 - probability), inf)
        if location == "middle": result = (normal_probability(mean, standard_deviation).ppf((1 - probability) / 2), normal_probability(mean, standard_deviation).ppf(1 - (1 - probability) / 2))
        if location == "ends": result = (normal_probability(mean, standard_deviation).ppf((probability) / 2), normal_probability(mean, standard_deviation).ppf(1 - (probability) / 2))
        interval = result
            
    if graph:
        plt.grid(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        axes1 = plt.gca()
        axes1.set_xlabel("standardized units")
        if mean != 0 or standard_deviation != 1:
            axes2 = axes1.twiny()
            axes2.set_xlabel("original units")
            axes1.set_xticks([-4.4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.4])
            #locs = axes1.xaxis.get_ticklocs()
            #axes1.set_xticks(locs)
            xticks = axes1.xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)
            xticks[0].tick1line.set_visible(False) 
            xticks[-1].tick1line.set_visible(False) 
        dots = 8 * standard_deviation / 1000 
        
        x = numpy.linspace(mean - 4 * standard_deviation, mean + 4 * standard_deviation, 1000)
        plt.plot(x, normal_probability(mean, standard_deviation).pdf(x), 'b')
    
        shade = list(interval)
        if shade[0] < mean - 4 * standard_deviation: shade[0] = mean - 4 * standard_deviation
        if shade[1] > mean + 4 * standard_deviation: shade[1] = mean + 4 * standard_deviation
        if location != "ends":
            plt.fill_between(numpy.arange(shade[0], shade[1], dots), normal_probability(mean, standard_deviation).pdf(numpy.arange(shade[0], shade[1], dots)), color='r')
        else:
            plt.fill_between(numpy.arange(mean - 4 * standard_deviation, shade[0], dots), normal_probability(mean, standard_deviation).pdf(numpy.arange(mean - 4 * standard_deviation, shade[0], dots)), color='r')
            plt.fill_between(numpy.arange(shade[1], mean + 4 * standard_deviation, dots), normal_probability(mean, standard_deviation).pdf(numpy.arange(shade[1], mean + 4 * standard_deviation, dots)), color='r')
        for value in graph_values:
            if value > mean - 4 * standard_deviation and value < mean + 4 * standard_deviation:
                plt.plot(numpy.array([value, value]), numpy.array([0, normal_probability(mean, standard_deviation).pdf(value)]), color='g')    
       
        plt.show()
    return result

def t(interval = None, mean = 0, standard_deviation = 1, scale = None, degrees_of_freedom = None, probability = None, location = "left", graph = False, graph_values = [], ret = None): 
    if scale != None: standard_deviation = scale
    if standard_deviation < 0: standard_deviation = -standard_deviation
    if ret == "probability":
        if type(interval) != tuple and type(interval) != list:
            if location == "left": interval = (-inf, interval)
            elif location == "right": interval = (interval, inf)
            else:
                if interval > mean: interval = (mean - abs(interval - mean), interval)
                else: interval = (interval, mean + abs(interval - mean))
        
        if interval[0] > interval[1]: interval = [interval[1], interval[0]]
        result = t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).cdf(interval[1]) - t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).cdf(interval[0])
        if location == "ends": result = 1 - result
        probability = result
    
    #if inspect.getouterframes(inspect.currentframe(), 2)[1][3] == "score":
    if ret == "interval":
        if location == "left": result = (-inf, t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(probability))
        if location == "right": result = (t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - probability), inf)
        if location == "middle": result = (t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf((1 - probability) / 2), t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - (1 - probability) / 2))
        if location == "ends": result = (t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf((probability) / 2), t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - (probability) / 2))
        interval = result
            
    if graph:
        plt.grid(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        axes1 = plt.gca()
        axes1.set_xlabel("standardized units")
        if mean != 0 or standard_deviation != 1:
            axes2 = axes1.twiny()
            axes2.set_xlabel("original units")
            axes1.set_xticks([-4.4, -4, -3, -2, -1, 0, 1, 2, 3, 4, 4.4])
            #locs = axes1.xaxis.get_ticklocs()
            #axes1.set_xticks(locs)
            xticks = axes1.xaxis.get_major_ticks()
            xticks[0].label1.set_visible(False)
            xticks[-1].label1.set_visible(False)
            xticks[0].tick1line.set_visible(False) 
            xticks[-1].tick1line.set_visible(False) 
        dots = 8 * standard_deviation / 1000
        
        x = numpy.linspace(mean - 4 * standard_deviation, mean + 4 * standard_deviation, 1000)
        plt.plot(x, normal_probability(mean, standard_deviation).pdf(x), 'y')
        
        x = numpy.linspace(mean - 4 * standard_deviation, mean + 4 * standard_deviation, 1000)
        plt.plot(x, t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(x), 'b')
    
        shade = list(interval)
        if shade[0] < mean - 4 * standard_deviation: shade[0] = mean - 4 * standard_deviation
        if shade[1] > mean + 4 * standard_deviation: shade[1] = mean + 4 * standard_deviation
        if location != "ends":
            plt.fill_between(numpy.arange(shade[0], shade[1], dots), t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(shade[0], shade[1], dots)), color='r')
        else:
            plt.fill_between(numpy.arange(mean - 4 * standard_deviation, shade[0], dots), t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(mean - 4 * standard_deviation, shade[0], dots)), color='r')
            plt.fill_between(numpy.arange(shade[1], mean + 4 * standard_deviation, dots), t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(shade[1], mean + 4 * standard_deviation, dots)), color='r')
        for value in graph_values:
            if value > mean - 4 * standard_deviation and value < mean + 4 * standard_deviation:
                plt.plot(numpy.array([value, value]), numpy.array([0, t_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(value)]), color='g')
        plt.show()
    return result

def chi2(interval = None, mean = 0, standard_deviation = 1, scale = None, degrees_of_freedom = None, probability = None, location = "right", graph = False, graph_values = [], ret = None):         
    if scale != None: standard_deviation = scale
    if standard_deviation < 0: standard_deviation = -standard_deviation
    if ret == "probability":
        if type(interval) != tuple and type(interval) != list:
            if location == "left": interval = (-inf, interval)
            elif location == "right": interval = (interval, inf)
            else:
                if interval > mean: interval = (mean - abs(interval - mean), interval)
                else: interval = (interval, mean + abs(interval - mean))
        
        if interval[0] > interval[1]: interval = [interval[1], interval[0]]
        result = chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).cdf(interval[1]) - chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).cdf(interval[0])
        if location == "ends": result = 1 - result
        probability = result
    
    #if inspect.getouterframes(inspect.currentframe(), 2)[1][3] == "score":
    if ret == "interval":
        if location == "left": result = (-inf, chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(probability))
        if location == "right": result = (chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - probability), inf)
        if location == "middle": result = (chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf((1 - probability) / 2), chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - (1 - probability) / 2))
        if location == "ends": result = (chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf((probability) / 2), chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).ppf(1 - (probability) / 2))
        interval = result
            
    if graph:
        plt.grid(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        axes1 = plt.gca()
        axes1.set_xlabel("standardized units")
        
        dots = 8 * standard_deviation / 1000
        
        x = numpy.linspace(chi_square_probability.ppf(0.00001, degrees_of_freedom), chi_square_probability.ppf(0.999, degrees_of_freedom), 1000)
        plt.plot(x, chi_square_probability.pdf(x, degrees_of_freedom), 'b')
        
        shade = list(interval)
        if shade[0] < chi_square_probability.ppf(0.00001, degrees_of_freedom): shade[0] = chi_square_probability.ppf(0.00001, degrees_of_freedom)
        if shade[1] > chi_square_probability.ppf(0.999, degrees_of_freedom): shade[1] = chi_square_probability.ppf(0.999, degrees_of_freedom)
        if location != "ends":
            plt.fill_between(numpy.arange(shade[0], shade[1], dots), chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(shade[0], shade[1], dots)), color='r')
        else:
            plt.fill_between(numpy.arange(chi_square_probability.ppf(0.00001, degrees_of_freedom), shade[0], dots), chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(chi_square_probability.ppf(0.00001, degrees_of_freedom), shade[0], dots)), color='r')
            plt.fill_between(numpy.arange(shade[1], chi_square_probability.ppf(0.999, degrees_of_freedom), dots), chi_square_probability(loc=mean, scale=standard_deviation, df=degrees_of_freedom).pdf(numpy.arange(shade[1], chi_square_probability.ppf(0.999, degrees_of_freedom), dots)), color='r')
        
        
        for value in graph_values:
            if value > chi_square_probability.ppf(0.00001, degrees_of_freedom) and value < chi_square_probability.ppf(0.999, degrees_of_freedom):
                plt.plot(numpy.array([value, value]), numpy.array([0, chi_square_probability(degrees_of_freedom).pdf(value)]), color='g')
        
        plt.show()
        
    return result

def f(interval = None, mean = 0, standard_deviation = 1, scale = None, degrees_of_freedom_1 = None, degrees_of_freedom_2 = None, probability = None, location = "right", graph = False, graph_values = [], ret = None):         
    if scale != None: standard_deviation = scale
    if standard_deviation < 0: standard_deviation = -standard_deviation
    if ret == "probability":
        if type(interval) != tuple and type(interval) != list:
            if location == "left": interval = (-inf, interval)
            elif location == "right": interval = (interval, inf)
            else:
                if interval > mean: interval = (mean - abs(interval - mean), interval)
                else: interval = (interval, mean + abs(interval - mean))
        
        if interval[0] > interval[1]: interval = [interval[1], interval[0]]
        result = f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).cdf(interval[1]) - f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).cdf(interval[0])
        if location == "ends": result = 1 - result
        probability = result
    
    #if inspect.getouterframes(inspect.currentframe(), 2)[1][3] == "score":
    if ret == "interval":
        if location == "left": result = (-inf, f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf(probability))
        if location == "right": result = (f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf(1 - probability), inf)
        if location == "middle": result = (f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf((1 - probability) / 2), f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf(1 - (1 - probability) / 2))
        if location == "ends": result = (f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf((probability) / 2), f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).ppf(1 - (probability) / 2))
        interval = result
            
    if graph:
        plt.grid(False)
        plt.gca().axes.yaxis.set_visible(False)
        plt.gca().axes.yaxis.set_ticks([])
        axes1 = plt.gca()
        axes1.set_xlabel("standardized units")
        
        dots = 8 * standard_deviation / 1000
        
        x = numpy.linspace(f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), 1000)
        plt.plot(x, f_probability.pdf(x, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), 'b')
        
        shade = list(interval)
        if shade[0] < f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2): shade[0] = f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2)
        if shade[1] > f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2): shade[1] = f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2)
        if location != "ends":
            plt.fill_between(numpy.arange(shade[0], shade[1], dots), f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).pdf(numpy.arange(shade[0], shade[1], dots)), color='r')
        else:
            plt.fill_between(numpy.arange(f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), shade[0], dots), f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).pdf(numpy.arange(f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), shade[0], dots)), color='r')
            plt.fill_between(numpy.arange(shade[1], f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), dots), f_probability(loc=mean, scale=standard_deviation, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).pdf(numpy.arange(shade[1], f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2), dots)), color='r')
        
        
        for value in graph_values:
            if value > f_probability.ppf(0.00001, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2) and value < f_probability.ppf(0.999, dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2):
                plt.plot(numpy.array([value, value]), numpy.array([0, f_probability(dfn=degrees_of_freedom_1, dfd = degrees_of_freedom_2).pdf(value)]), color='g')
        
        plt.show()
        
    return result

def linear_regression(x_col, y_col, show = False, return_slope_information = True):
    
    global data
    
    data_copy = data[[x_col] + [y_col]].dropna()
    
    x = data_copy[x_col].values
    y = data_copy[y_col].values
    n = len(y)
    b1, b0, r, p, se = linregress(data_copy[x_col], data_copy[y_col])
    b1 = float(b1)
    se = float(se)
    s_x = data_copy[x_col].std()
    s_y = data_copy[y_col].std()
    y_mean = y.mean()
    y_pred = [b0 + b1 * x_val for x_val in x]
    residuals = y - y_pred
    deviations = [y_value - y_mean for y_value in y]
    reg = [y_value - y_mean for y_value in y_pred]
    
    SSE = sum(map(lambda i : i * i, residuals))
    SER = sqrt(SSE/(n-2))
    SST = sum(map(lambda i : i * i, deviations))
    SSR = sum(map(lambda i : i * i, reg))
    
    if show:
        print("Scatterplot of data and regression line:")
        #matplotlib.pyplot.clf()
        data_copy.plot.scatter(x=x_col,y=y_col)
        plt.plot(x, b0 + b1 * x, color="Red")
        plt.show()

        print("Scatterplot of residuals vs. predicted values:")
        data2 = pandas.DataFrame({"predicted values":y_pred, "residuals":residuals})
        data2.plot.scatter(x="predicted values",y="residuals")
        plt.show()

        print("Normal probability plot of residuals:")
        probplot(residuals, dist="norm", plot=plt) 
        plt.show()

        print("regression equation coefficients:")
        print("    b0:", "{:.4f}".format(b0))
        print("    b1:", "{:.4f}".format(b1))
        print("p-value (b1 != 0):", "{:.4f}".format(p))
        print("r-squared:", "{:.4f}".format(r ** 2))
        print("correlation coefficient (r):", "{:.4f}".format(r))
        print("standard error of the residuals (estimate) (SER or s):", "{:.4f}".format(SER))
        print("standard error of slope (se):", "{:.4f}".format(se))
        print("sum of squares total (SST): ", "{:.4f}".format(SST))
        print("sum of squares error (SSE): ", "{:.4f}".format(SSE))
        print("sum of squares due to regression (SSR): ", "{:.4f}".format(SSR))
        print()
        print("predicted values:", truncate(lambda:y_pred)())
        print("residuals:", truncate(lambda:list(residuals))())
        
    if return_slope_information: return b1, se    
    else: return b1 / se

def multiple_regression(x_cols, y_col, show = False):
    global data
    if type(x_cols) == str: x_cols = [] + [x_cols]
    if type(x_cols) != list: x_cols = list(x_cols)
    
    data["intercept"] = 1
    x_cols = ["intercept"] + x_cols
    
    data_copy = data[x_cols + [y_col]].dropna()
    
    x_full = data_copy[x_cols].values
    y = data_copy[y_col].values
    
    x_cols = x_cols[1:]
    
    x = data_copy[x_cols].values
    y = data_copy[y_col].values
    k = len(x_cols)
    n = len(y)
    regression = linear_model.LinearRegression()
    regression.fit(x, y)
    b = [ regression.intercept_ ]
    b.extend( regression.coef_ )
    
    
    
    y_pred = regression.predict(x)
    residuals = y - y_pred
    x_mean = [ data_copy[x_col].mean() for x_col in x_cols ]
    y_mean = y.mean()
    deviations = [y_value - y_mean for y_value in y]
    reg = [y_value - y_mean for y_value in y_pred]
    SSE = sum(map(lambda i : i * i, residuals))
    SST = sum(map(lambda i : i * i, deviations))
    SSR = sum(map(lambda i : i * i, reg))
    
    MSR = SSR / k
    MSE = SSE / (n - k - 1)
    SER = sqrt(MSE)
    r_squared = SSR / SST
    adj_r_squared = 1 - (1 - r_squared) * ((n - 1) / (n - k - 1))
    F_data = MSR / MSE
    df_1 = k
    df_2 = n - k - 1
    p_value = 1 - f_probability.cdf(F_data, df_1, df_2)
    
    if show:
        print("Scatterplot of residuals vs. predicted values:")
        data2 = pandas.DataFrame({"predicted values":y_pred, "residuals":residuals})
        data2.plot.scatter(x="predicted values",y="residuals")
        plt.plot([min(y_pred),max(y_pred)], [0,0], color="Red")
        plt.show()

        print("Normal probability plot of residuals:")
        probplot(residuals, dist="norm", plot=plt) 
        plt.show()

        print("Regression equation:")
        print(y_col,"=","{:.4f}".format(b[0]),end="")
        for i in range(0,k): print(" + ", "{:.4f}".format(b[i + 1]), "(", x_cols[i], ")",sep="",end="")
        print()
        
        diagonal = numpy.diagonal(MSE * numpy.linalg.inv(numpy.dot(x_full.T, x_full))).tolist()
        for i in range(len(diagonal)):
            if diagonal[i] < 0: diagonal[i] = 0
        diagonal = numpy.array(diagonal)        
                
        SE_x = pandas.Series(numpy.sqrt(diagonal))
        t_x = b / SE_x
        p_x = 2 * (1 - t_probability.cdf(numpy.abs(t_x), n - k))  
        individual_predictors = pandas.DataFrame({"Coef":truncate(lambda:b)(), "SE Coef":truncate(lambda:SE_x)(), "t":truncate(lambda:t_x)(), "p":truncate(lambda:p_x)()})
        individual_predictors.index = ["Constant"] + x_cols
        print(individual_predictors, flush=True)

        print("standard error of the residuals (estimate) (SER or s):", "{:.4f}".format(SER))
        print("r-squared:", "{:.4f}".format(r_squared))
        print("adjusted r-squared:", "{:.4f}".format(adj_r_squared))
        print("n:", n)
        print()

        print("Analysis of Variance:")
        print("sum of squares due to regression (SSR): ", "{:.4f}".format(SSR))
        print("sum of squares error (SSE): ", "{:.4f}".format(SSE))
        print("sum of squares total (SST): ", "{:.4f}".format(SST))
        print("mean square regression (MSR):", "{:.4f}".format(MSR))
        print("mean square error (MSE):", "{:.4f}".format(MSE))
        print("F_data:", "{:.4f}".format(F_data))
        print("p-value:", "{:.4f}".format(p_value))
        
        """
        print(data)
        
        data_backup = data.copy()
        data = pandas.DataFrame()
        values = list(chain(*list(data_backup[col].values) for col in x_cols))
        print(values)
        data["x"] = pandas.Series(values) 
        
        print(data)
        
        for i in range(k - 1):
            values = [0] * (n * k)
            for j in range(n):
                values[n * i + j] = 1
            data["t" + str(i + 1)] = values
        print("-----")
        print(data)
        print("-----")
        perform(anova, show = True)
        data = data_backup
        """
    
    return F_data

def interval_at_value(x_col, y_col, value, probability = 0.95, interval_type = None):
    global data
        
    data_copy = data[[x_col] + [y_col]].dropna()
    
    x = data_copy[x_col].values
    y = data_copy[y_col].values
    
    n = len(y)
    b1, b0, r, p, se = linregress(data_copy[x_col], data_copy[y_col])
    y_hat = b0 + b1 * value
    t_alpha_over_2 = t_probability(loc=0, scale=1, df=n-2).ppf((1 - probability) / 2)
    s_x = data[x_col].std()
    s_y = data[y_col].std()
    x_mean = x.mean()
    y_mean = y.mean()
    y_pred = [b0 + b1 * x_val for x_val in x]
    residuals = y - y_pred
    deviations = [y_value - y_mean for y_value in y]
    reg = [y_value - y_mean for y_value in y_pred]
    
    SSE =  sum(map(lambda i : i * i, residuals))
    SER = sqrt(SSE/(n-2))
    SST =  sum(map(lambda i : i * i, deviations))
    SSR = sum(map(lambda i : i * i, reg))
    
    if interval_type == "confidence": margin_of_error = t_alpha_over_2 * SER * sqrt(1 / n + (value - x_mean) ** 2 / sum([(x_i - x_mean) ** 2 for x_i in x ]))
    elif interval_type == "prediction": margin_of_error = t_alpha_over_2 * SER * sqrt(1 + 1 / n + (value - x_mean) ** 2 / sum([(x_i - x_mean) ** 2 for x_i in x ]))

    print("Lower bound:", truncate(lambda:y_hat - abs(margin_of_error))())
    print("Upper bound:", truncate(lambda:y_hat + abs(margin_of_error))())

population = False

data = None
data_bk = None
labels = None
indexes = None
index_label = None

probability_distribution = False
X = None
#P = None

gss_current = None
gss_historical = None

table = None
group_by = None
hist_group_by = None
append = False
bins = None
hist_bins = None
include = None
hide_empty = True
frequency = True
relative = False
hist_relative = None
cumulative = False
cumulative_relative = False
index_sort = False
show_nan = False
show_totals = False

graph = None
lines = False
x_column = None
y_column = None

print("All course code is now loaded!")