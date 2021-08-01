import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import re
import sys
sys.path.append('../process data/')
import scipy.stats as stats
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.offline as pyo
import plotly.express as px
from encode_processed_data import encode_data
from plotly.colors import n_colors



class summary_plots_and_figures:
    def __init__(self, ed):
        self.ed = ed
        self.wcst_performance = None

        # ============================================== Summary Table Data ==============================================
        self.continuous_vars = [
            {'label': 'N-Back accuracy',        'value': 'nback_status'},
            {'label': 'N-Back Reaction Time',   'value': 'nback_reaction_time_ms'},
            {'label': 'Fitts accuracy',         'value': 'fitts_mean_deviation'},
            {'label': 'Corsi Span',             'value': 'corsi_block_span'},
            {'label': 'Navon accuracy',         'value': 'navon_perc_correct'},
            {'label': 'Navon Reaction Time',    'value': 'navon_reaction_time_ms'},
            {'label': 'WCST accuracy',          'value': 'wcst_accuracy'},
            {'label': 'WCST Reaction Time',     'value': 'wcst_RT'},
            {'label': 'Age',                    'value': 'demographics_age_a'},
            {'label': 'Computer Hours',         'value': 'demographics_computer_hours_a'},
            {'label': 'Income',                 'value': 'demographics_income_a'},
            {'label': 'Initial Response RT',    'value': 'demographics_mean_reation_time_ms'}
        ]

        self.categorical_vars = [
            {'label': 'Gender',                 'value': 'demographics_gender_a'},
            {'label': 'Education',              'value': 'demographics_education_a'},
            {'label': 'Handedness',             'value': 'demographics_handedness_a'},
            {'label': 'Age',                    'value': 'demographics_age_group'},
            {'label': 'Navon Level',            'value': 'navon_level_of_target'},
            {'label': 'Nback',                  'value': 'nback_group'},
            {'label': 'Fitts',                  'value': 'fitts_group'},
            {'label': 'Corsi',                  'value': 'corsi_group'},
            {'label': 'Navon',                  'value': 'navon_group'},
            {'label': 'WCST',                   'value': 'wcst_group'},
            {'label': 'Random Particpants',     'value': 'random_participants'},
            {'label': 'None',                   'value': ''}
        ]
        # ============================================== Summary Table Data ==============================================


        # ============================================== Demographics ==============================================
        self.demographic_groups = [
            {'label': 'gender', 'value': 'gender_a'}, {'label': 'handedness', 'value': 'handedness_a'}, 
            {'label': 'education', 'value': 'education_a'}, {'label': 'age', 'value': 'age_group'}]

        self.demographic_cont_vars = [
            {'label': 'age', 'value': 'age_a'}, {'label': 'income', 'value': 'income_a'}, 
            {'label': 'computer_hours', 'value': 'computer_hours_a'}, {'label': 'reaction time (ms)', 'value': 'mean_reation_time_ms'}
        ]

        self.demo_pie_map = {
            'gender_a':     {'dummy_var':'gender_a',        'labels':['male', 'female', 'other'],                       'colors':['steelblue', 'darkred', 'cyan'],                                          'title':'Gender Distribution',     'name':'gender'},
            'education_a':  {'dummy_var':'education_a',     'labels':['university', 'graduate school', 'high school'],  'colors':['rgb(177, 127, 38)', 'rgb(129, 180, 179)', 'rgb(205, 152, 36)'],  'title':'Education Distribution',   'name':'education'},
            'handedness_a': {'dummy_var':'handedness_a',    'labels':['right', 'left', 'ambidextrous'],                 'colors':px.colors.sequential.RdBu,                                         'title':'Handedness Distribution',  'name':'handedness'},
            'age_group':    {'dummy_var':'age_group',       'labels':np.unique(ed.demographics[['age_group']]).tolist(),'colors':px.colors.sequential.GnBu,                                         'title':'Age Distribution',         'name':'age'}
            }
            
        self.demo_continuous_naming = {
            'age_a':                   {'xlab':'Age',                      'ylab':'Count', 'name':'Age Distribution by '},
            'income_a':                {'xlab':'Income',                   'ylab':'Count', 'name':'Income Distribution by '},
            'computer_hours_a':        {'xlab':'Computer hours',           'ylab':'Count', 'name':'Computer Hours Distribution by '},
            'mean_reation_time_ms':    {'xlab':'RT (reaction time (ms))',  'ylab':'Count', 'name':'RT Distribution by '},
        }
        # ============================================== Demographics ==============================================

    

    
    def available_data(self):
        message = """
            scatter_plot                : plot
            distribution_plot           : plot
            basic_pie_chart             : plot
            basic_distributional_plots  : plot
            compute_summary_stats       : table
            ANOVA                       : table
            """
        print(message)

    def create_performance_groupings(self, n_steps=10):
        """Compute Performance Bins (per task)"""
        aa = ['nback_group', 'fitts_group','corsi_group','navon_group','wcst_group']
        bb = ['nback_status', 'fitts_mean_deviation', 'corsi_block_span', 'navon_perc_correct', 'wcst_accuracy']
        spd=self.ed.summary_table

        for a, b in zip(aa, bb):
            # ---- groups
            srt=min(spd[b]); stp=max(spd[b]); 
            steps = np.linspace(start=srt, stop=stp, num=n_steps)
            grps  = [str(np.round(steps[i],2)) + '-' + str(np.round(steps[i+1],2)) for i in range(len(steps)-1)]

            spd[a] = 'Na'
            for s in range(n_steps-1):
                spd.loc[(spd[b]>steps[s]) & (spd[b]<=steps[s+1]),a] = grps[s]
        self.ed.summary_table = spd



    def compute_wcst_performance_trial_bins(self, n_bins=10):
        """Return: DataFrame capturing the performance per n_bins trials"""
        wcst_data=self.ed.raw.wcst_data

        # ---- add trial number ----x
        xx = []; df = wcst_data
        [xx.append((i%100)+1) for i in range(df.shape[0])]
        df['trial_no'] = xx 

        # ---- status==1 --> correct
        t = np.linspace(0,100,num=n_bins+1).tolist(); c=0

        for tt in t[1:]:
            c +=1
            x = df.loc[df['trial_no'] < tt,].groupby(['participant', 'status']).agg({
            'participant':              ['count'],
            'reaction_time_ms':         ['mean', 'std'],
            'perseverance_error':       ['mean'],
            'not_perseverance_error':   ['mean']
            }).reset_index()
            x['percentages'] = x[('participant', 'count')]/tt
            x['trials']      = str(round(t[c-1])) + '-' + str(round(t[c]))
            x['trials_2']    = t[c]
            if c==1:    data=x
            else:       data=data.append(other=x)

        # if x>0 --> perseverance_error > not_perseverance_error --> main error=perseverance_error
        data['main_error'] = np.where(data['perseverance_error'] - data['not_perseverance_error'] > 0, 'perserverance errors', 'non perserverance errors')
        self.wcst_performance = data
    
    #---- random sample of n participants ----x
    def random_participants_sample(self, n=10):
        spd=self.ed.summary_table
        participants = np.random.choice(spd.index.unique(), n)
        participants
        spd['random_participants'] = 'other'
        for p in participants:
            spd.loc[spd.index==p, 'random_participants'] = p

        self.ed.summary_table = spd


    def wcst_performance_plot_deprecated(self, 
        group='corsi_group', 
        mean_plot=False,
        colours=px.colors.sequential.Plasma,
        title='WCST Performance', 
        xaxis={'title':'trials'}, 
        yaxis={'title':'% Correct'}, 
        template='none', 
        legend_title_text='', width=900, height=500):

        # ---- fetch data ----x
        summary_data=self.ed.summary_table.copy()
        wcst_performance_data=self.wcst_performance.copy()

        # --- join data + compute performance per group ---x
        x = wcst_performance_data.set_index(('participant',''))
        x = x.loc[x['status']==1,:] #???
        data = x.join(summary_data)

        
        def compute_performance_per_group(data, group):
            x = data.groupby([group, ('trials_2', '')]).agg({
                ('percentages','')          :'mean', 
                ('reaction_time_ms','mean') :'mean'
            }).reset_index()
            x.columns = [xx[0] for xx in x.columns]
            return x
        if not group:
                data['all'] = 'all data' 
                group = 'all'
        data = compute_performance_per_group(data=data, group=group)


        # ---------- plots ----------x
        if not legend_title_text: legend_title_text = group
        groups = data[group].unique()
        traces = []
        c=-1
        for g in groups:
            c+=1
            if g != 'Na' and g != 'other':
                df    = data.loc[(data[group] == g), ['trials_2', 'percentages', 'reaction_time_ms']]
                trace = go.Scatter(x=df.trials_2, y=df.percentages, mode='lines+markers', name='{}'.format(g),
                        line=dict(color='black'), 
                        marker=dict(
                            size=df['reaction_time_ms']/100,
                            color=colours[c],
                            opacity=0.75,
                            line=dict(color='white')))
                traces.append(trace)
        
        if mean_plot:
            df = data.groupby('trials_2').agg({
                'reaction_time_ms': ['mean', 'std'],
                'percentages':      ['mean', 'std']}).reset_index()
            g  = 'aggregate'
            # df['participant'] = g
            trace = go.Scatter(x=df.trials_2, y=df[('percentages','mean')], mode='lines+markers', name='{}'.format(g),
                    line=dict(color='black'), 
                    marker=dict(
                        size=df[('reaction_time_ms','mean')]/100,
                        color=colours[-1],
                        opacity=0.75,
                        line=dict(color='white')))
            traces.append(trace)


        layout  = go.Layout(title=title, xaxis=xaxis, yaxis=yaxis, template=template, legend_title_text=legend_title_text, width=width, height=height)
        fig     = go.Figure(data=traces, layout=layout)
        return {'figure': fig, 'data':data}


    def wcst_performance_plot(
        self, group='corsi_group', 
        mean_plot=False,
        colours=px.colors.sequential.Plasma,
        title='WCST Performance', 
        xaxis={'title':'trials'}, 
        yaxis={'title':'% Correct'}, 
        template='none', 
        legend_title_text='', width=900, height=500):

        # ---- fetch data ----x
        summary_data=self.ed.summary_table.copy()
        wcst_performance_data=self.wcst_performance.copy()

        # --- join data + compute performance per group ---x
        x = wcst_performance_data.set_index(('participant',''))
        x = x.loc[x['status']==1,:] #???
        data = x.join(summary_data)

        # # ---- reset column names ----x
        cols = []
        for c in data.columns:
                if type(c) is tuple: cols.append('_'.join(c).strip('_'))
                else: cols.append(c)
        data.columns = cols
        
        def compute_performance_per_group(data, group):
            x = data.groupby([group, 'trials_2']).agg({
                'percentages'           :'mean', 
                'reaction_time_ms_mean' :'mean'
                }).reset_index()
            # x.columns = [xx[0] for xx in x.columns]
            return x
        if not group:
                data['all'] = 'all data' 
                group = 'all'
        data = compute_performance_per_group(data=data, group=group)


        # ---------- plots ----------x
        if not legend_title_text: legend_title_text = group
        groups = data[group].unique()
        traces = []
        c=-1
        for g in groups:
            c+=1
            if g != 'Na' and g != 'other':
                df    = data.loc[(data[group] == g), ['trials_2', 'percentages', 'reaction_time_ms_mean']]
                trace = go.Scatter(x=df.trials_2, y=df.percentages, mode='lines+markers', name='{}'.format(g),
                        line=dict(color='black'), 
                        marker=dict(
                            size=df['reaction_time_ms_mean']/100,
                            color=colours[c],
                            opacity=0.75,
                            line=dict(color='white')))
                traces.append(trace)
        
        if mean_plot:
            print('----------------------')
            print(df.head())
            df = data.groupby('trials_2').agg({
                'percentages'           :'mean', 
                'reaction_time_ms_mean' :'mean'
                }).reset_index()
            g  = 'aggregate'
            trace = go.Scatter(x=df.trials_2, y=df['percentages'], mode='lines+markers', name='{}'.format(g),
                    line=dict(color='black'), 
                    marker=dict(
                        size=df['reaction_time_ms_mean']/100,
                        color=colours[-1],
                        opacity=0.75,
                        line=dict(color='white'))
                        )
            traces.append(trace)


        layout  = go.Layout(title=title, xaxis=xaxis, yaxis=yaxis, template=template, legend_title_text=legend_title_text, width=width, height=height)
        fig     = go.Figure(data=traces, layout=layout)
        return {'figure': fig, 'data':data}
    


# group_var='corsi_group'
# group_var='random_participants'
# data = wcst_performance(group=group_var, mean_plot=True)
# data['figure']


    def scatter_plot(self, data, xvar, yvar, group_var=False, xlab='', ylab='', title='', cols=sum([px.colors.sequential.Plasma*3], [])):

        if not group_var: 
            data = data[[xvar, yvar]].dropna()
            traces = [go.Scatter(x=data[xvar], y=data[yvar], mode='markers', marker_color=cols[0])]
            layout = go.Layout( title=title, xaxis={'title':xlab}, yaxis={'title':ylab}, template='none')
            legend_title_text=group_var
        else:
            legend_title_text=''
            data = data[[xvar, yvar, group_var]].dropna()
            traces = []; c=0
            for g in data[group_var].unique():
                c += 1
                dt = data.loc[data[group_var]==g,]
                traces.append(go.Scatter(x=dt[xvar], y=dt[yvar], mode='markers', marker_color=cols[c], name=g))
            layout = go.Layout( title=title, xaxis={'title':xlab}, yaxis={'title':ylab}, template='none', legend_title_text=legend_title_text)
        fig = go.Figure(data=traces, layout=layout)
        return fig


    # cols=['#A56CC1', '#A6ACEC', '#63F5EF', 'steelblue', 'darkblue', 'blue', 'darkred', '#756384']
    def distribution_plot(self, data, xvar, nbinsx=10, opacity=1, group_var=False, xlab='', ylab='', title='', cols= px.colors.sequential.Plasma):
        """Distribution of Variable 1"""
        if title=='': 
            if group_var: title = 'Distribution of ' + str(xvar) + ' by ' + str(group_var)
            else: title = 'Distribution of ' + str(xvar)

        if not group_var: 
            data   = data[[xvar]].dropna()
            traces = [go.Histogram(x=data[xvar], marker_color=cols[0], nbinsx=nbinsx, opacity=opacity)]
            layout = go.Layout(title=title, xaxis={'title':xlab}, yaxis={'title':ylab}, template='none')
            fig    = go.Figure(data=traces, layout=layout)
            return fig
        else:
            fig = make_subplots(rows=2, cols=1, subplot_titles=('', ''))
            data   = data[[xvar, group_var]].dropna()
            traces = []; c=0; RTs = []
            for g in data[group_var].unique():
                dt = data.loc[data[group_var]==g,]
                RTs.append(dt[xvar])
                fig.add_trace(go.Histogram(x=dt[xvar], nbinsx=nbinsx, marker_color=cols[c], name=g, opacity=opacity), row=2, col=1)
                c += 1

            # ---- sort lists ----x
            srt = np.argsort([np.mean(r) for r in RTs])
            RT  = [RTs[s] for s in srt]

            # ---- create figure: violin plots ----x
            c=-1
            for nm, rt in zip(data[group_var].unique(), RT):
                c+=1
                fig.add_trace(go.Violin(
                    showlegend=False, y=rt, name=nm, box_visible=True,
                    meanline_visible=True, fillcolor=cols[c], line_color=cols[-1]), row=1, col=1)
            
            
            fig.update_layout(title_text=title, height=700, template='none')
            return fig


    def ANOVA(self, data, group_var, value_var):
        data[data.columns[data.dtypes=='string']]=data[data.columns[data.dtypes=='string']].astype(object)
        if not group_var:
            group_var = 'demographics_handedness_a'
        # Create ANOVA backbone table
        raw_data = [['Between Groups', '', '', '', '', '', ''], ['Within Groups', '', '', '', '', '', ''], ['Total', '', '', '', '', '', '']] 
        anova_table = pd.DataFrame(raw_data, columns = ['Source of Variation', 'SS', 'df', 'MS', 'F', 'P-value', 'F crit']) 
        anova_table.set_index('Source of Variation', inplace = True)

        # calculate SSTR and update anova table
        x_bar = data[value_var].mean()
        SSTR = data.groupby(group_var).count() * (data.groupby(group_var).mean() - x_bar)**2
        anova_table['SS']['Between Groups'] = SSTR[value_var].sum()

        # calculate SSE and update anova table
        SSE = (data.groupby(group_var).count() - 1) * data.groupby(group_var).std()**2
        anova_table['SS']['Within Groups'] = SSE[value_var].sum()

        # calculate SSTR and update anova table
        SSTR = SSTR[value_var].sum() + SSE[value_var].sum()
        anova_table['SS']['Total'] = SSTR

        # update degree of freedom
        anova_table['df']['Between Groups'] = data[group_var].nunique() - 1
        anova_table['df']['Within Groups'] = data.shape[0] - data[group_var].nunique()
        anova_table['df']['Total'] = data.shape[0] - 1

        # calculate MS
        anova_table['MS'] = anova_table['SS'] / anova_table['df']

        # calculate F 
        F = anova_table['MS']['Between Groups'] / anova_table['MS']['Within Groups']
        anova_table['F']['Between Groups'] = F

        # p-value
        anova_table['P-value']['Between Groups'] = 1 - stats.f.cdf(F, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

        # F critical 
        alpha = 0.05
        # possible types "right-tailed, left-tailed, two-tailed"
        tail_hypothesis_type = "two-tailed"
        if tail_hypothesis_type == "two-tailed":
            alpha /= 2
        anova_table['F crit']['Between Groups'] = stats.f.ppf(1-alpha, anova_table['df']['Between Groups'], anova_table['df']['Within Groups'])

        return anova_table.reset_index()
    

    def compute_summary_stats(self, data, value_var='wcst_RT', group_var='demographics_education_a', resetIndex=False):
        data = data.dropna()
        if not group_var:
            data['data'] = 'all data' 
            group_var = 'data'
        x = data.groupby(group_var).agg({
            'wcst_accuracy':            ['mean', 'std'],
            'wcst_RT':                  'mean',
            'navon_perc_correct':       ['mean', 'std'],
            'navon_reaction_time_ms':   'mean',
            'nback_status':             ['mean', 'std'],
            'nback_reaction_time_ms':   'mean',
            'fitts_mean_deviation':     ['mean', 'std'],
            'corsi_block_span':         ['mean', 'std']    
            })
        if resetIndex:
            new_cols = [(a + ' ' + b) for a,b in x.columns]
            x.columns = new_cols
            x = x.reset_index()
            return x.round(2)
        else: return x.round(2)


    def basic_pie_chart(self, dummy_var, labels, colors, title, df):
        if not dummy_var: dummy_var='dummy_var'; df[dummy_var] = 'all'
        if not labels: labels=df[dummy_var].unique()
        sub    = df[[dummy_var]].value_counts()
        values = sub.tolist()
        labels =  [x for x in labels if str(x) != '<NA>']
        print(labels)
        fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
        fig.update_traces(textfont_size=15, marker=dict(colors=colors, line=dict(color='white', width=0)))
        fig.update(layout_title_text=title)
        fig.update_layout(showlegend=False)
        return fig
        
        
    def basic_distributional_plots(self, group_var, continuous_var, xlab, ylab, title, df):
    
        group_var = self.demo_pie_map[group_var]
        fig = go.Figure()
        for c in range(len(group_var['labels'])):
            fig.add_trace(go.Histogram(
                x           =   df[continuous_var][df[group_var['dummy_var']] == group_var['labels'][c]],            
                # histnorm  ='percent',
                name        = group_var['labels'][c], 
                marker_color= group_var['colors'][c],
                opacity     = 1
            ))
        fig.update_layout(
            barmode         = 'overlay',
            title_text      = title, 
            xaxis_title_text= xlab, 
            yaxis_title_text= ylab, 
            bargap          = 0.05, 
            bargroupgap     = 0.1 
        )
        fig.update_layout(barmode='group', template='none')
        return fig
    

    




    
