def fetch_configuration():
    
    
    variables_jet = [
                                       
                                        'jet1_mass',
                                        'jet2_mass',
                                        'jet1_pt',
                                        'jet2_pt',
                                        'jet1_eta',
                                        'jet2_eta',
                                        'jet1_phi',
                                        'jet2_phi',
                                        'jet1btag',
                                        'jet2btag',
                                        
                                        
                               ]



    variables_dijet = [ 
                                     'Higgs_pt',
                                     'Higgs_eta',
                                     'Higgs_mass',
                                     'Higgs_phi',
                                     'btagged',
                                     'jet1btag',                                     
                                     'jet2btag',



                      ]
    variables_ak8 = [ 
                                     'Higgs_pt',
                                     'Higgs_eta',
                                     'Higgs_mass',
                                     'Higgs_phi',
                                     'btagged',
                 #                    'jet1btag',                                     
                  #                   'jet2btag',



                                     ]


    variables_score = [

                             'jet1_mass', #0
                             'jet2_mass', #1
                             'jet1_pt',#2
                             'jet2_pt',#3
                             'jet1_eta',#4
                             'jet2_eta',#5
                             'jet1_phi',#6
                             'jet2_phi',#7
                             'jet1btag',#8
                             'jet2btag',#9
                             'Higgs_pt',#10
                             'Higgs_eta',#11
                             'Higgs_mass',#12
                             'Higgs_phi',
                             'btagged',
                             #'MLscore',
                             ]


    variables_all = [
        
                             'jet1_mass', #0
                             'jet2_mass', #1
                             'jet1_pt',#2
                             'jet2_pt',#3
                             'jet1_eta',#4
                             'jet2_eta',#5
                             'jet1_phi',#6
                             'jet2_phi',#7
                             'jet1btag',#8
                             'jet2btag',#9
                             'Higgs_pt',#10
                             'Higgs_eta',#11
                             'Higgs_mass',#12
                             'Higgs_phi',
                             'btagged',
                             

                             ]
    variables_multi = [
        
                             'jet1_mass', #0
                             'jet2_mass', #1
                             'jet1_pt',#2
                             'jet2_pt',#3
                             'jet1_eta',#4
                             'jet2_eta',#5
                             'jet1_phi',#6
                             'jet2_phi',#7
                             'jet1btag',#8
                             'jet2btag',#9
                             'Higgs_pt',#10
                             'Higgs_eta',#11
                             'Higgs_mass',#12
                             'btagged',
                             'DeepB',
                             'delta_pt',
                             'delta_mass',
                             'delta_r',
                             'delta_eta',
                             'delta_phi',
                             

                             ]

    dic = {
        'O250':{


            'presel': '(jet1_pt>30 and jet2_pt>30)',  
            'bin' : 'Higgs_resolved==1',
            'bin':  'Higgs_pt>250',
            #'bin':
            'variables': variables_all, 

            },
        'U250':{


            'presel': '(jet1_pt>30 and jet2_pt>30)',
            'bin' : 'Higgs_resolved==1',
            'bin': 'Higgs_pt<=250',
            'bin':  'Higgs_pt>=100',
            'variables': variables_all,

            },
        '100':{

              'presel': '(jet1_pt>30 and jet2_pt>30)',
              'bin' : 'Higgs_resolved==1',
              'bin': 'Higgs_pt<=100',
              'variables': variables_all,

              },
       'MO250':{
            'enable_merged' : False, #ONLY RESOLVED
            'enable_resolved' : True,
            'file_name': './candidati1200.root',
            'presel': '(jet1_pt>30 and jet2_pt>30 or (jet1_pt==-1))',  
           
            'bin':  'Higgs_pt>250',
            #'bin':
            'variables': variables_multi, 
            'threshold_merged' : .5,
            'threshold_resolved' : .2,
            'n_estimators_resolved' : 130,
            'max_depth_resolved' : 4,
            'min_child_weight_resolved' : 4,
            'reg_alpha_resolved' : .01,
            },
        'MU250':{
            'enable_merged' : False, #ONLY RESOLVED
            'enable_resolved' : True,
            'file_name': './candidati1200.root',
            'presel': '(jet1_pt>30 and jet2_pt>30 or (jet1_pt==-1))',
           
            'bin': 'Higgs_pt<=250',
            'bin':  'Higgs_pt>=100',
            'variables': variables_multi,
            'threshold_merged' : .5,
            'threshold_resolved' : .2,
            'n_estimators_resolved' : 130,
            'max_depth_resolved' : 4,
            'min_child_weight_resolved' : 4,
            'reg_alpha_resolved' : .01,
            },
        'M100':{
              'enable_merged' : False, #ONLY RESOLVED
              'enable_resolved' : True,
              'file_name': './candidati1200.root',
              'presel': '(jet1_pt>30 and jet2_pt>30 or (jet1_pt==-1))',
           
              'bin': 'Higgs_pt<=100',
              #'bin':  'Higgs_pt>=100',
              'variables': variables_multi,
              'threshold_merged' : .5,
              'threshold_resolved' : .2,
              'n_estimators_resolved' : 130,
              'max_depth_resolved' : 4,
              'min_child_weight_resolved' : 4,
              'reg_alpha_resolved' : .01,
              },
        'FO250':{
            'enable_merged' : False, #ONLY RESOLVED
            'enable_resolved' : True,
            'file_name': './candidati1200.root',
            'presel': '(Higgs_pt>30)',  
            'bin' : 'Higgs_merged==1',
            'bin':  'Higgs_pt>250',
            #'bin':
            'variables': variables_ak8, 
            'threshold_merged' : .5,
            'threshold_resolved' : .2,
            'n_estimators_resolved' : 130,
            'max_depth_resolved' : 4,
            'min_child_weight_resolved' : 4,
            'reg_alpha_resolved' : .01,
            },
        'FU250':{
            'enable_merged' : False, #ONLY RESOLVED
            'enable_resolved' : True,
            'file_name': './candidati1200.root',
            'presel': '(Higgs_pt>30)',
            'bin' : 'Higgs_merged==1',
            'bin': 'Higgs_pt<=250',
            'bin':  'Higgs_pt>=100',
            'variables': variables_ak8,
            'threshold_merged' : .5,
            'threshold_resolved' : .2,
            'n_estimators_resolved' : 130,
            'max_depth_resolved' : 4,
            'min_child_weight_resolved' : 4,
            'reg_alpha_resolved' : .01,
            },
        'F100':{
              'enable_merged' : False, #ONLY RESOLVED
              'enable_resolved' : True,
              'file_name': './candidati1200.root',
              'presel': '(Higgs_pt>30)',
              'bin' : 'Higgs_merged==1',
              'bin': 'Higgs_pt<=100',
              #'bin':  'Higgs_pt>=100',
              'variables': variables_ak8,
              'threshold_merged' : .5,
              'threshold_resolved' : .2,
              'n_estimators_resolved' : 130,
              'max_depth_resolved' : 4,
              'min_child_weight_resolved' : 4,
              'reg_alpha_resolved' : .01,
              },
        }

    return dic 
