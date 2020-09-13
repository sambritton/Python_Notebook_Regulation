# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 13:54:40 2019

@author: samuel_britton
"""

import numpy as np
import pandas as pd
import random

from scipy.optimize import least_squares

def calculate_rate_constants(log_counts, rxn_flux,KQ_inverse, R, E_Regulation):
    KQ = np.power(KQ_inverse,-1)
    #Infer rate constants from reaction flux
    denominator = E_Regulation* np.exp(-R.dot(log_counts))*(1-KQ_inverse)
    # A reaction near equilibrium is problematic because (1-KQ_inverse)->0
    # By setting these reactions to be 
    # rate constant = 1/product_concs we are setting the rate to 1, which
    # is the same as the thermodynammic rate = KQ.
    one_idx, = np.where(KQ_inverse > 0.9)
    denominator[one_idx] = E_Regulation[one_idx]* np.exp(-R[one_idx,:].dot(log_counts));
    rxn_flux[one_idx] = 1;
    fwd_rate_constants = rxn_flux/denominator;
    
    return(fwd_rate_constants)

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def entropy_production_rate(KQ_f, KQ_r, E_Regulation):

    KQ_f_reg = E_Regulation * KQ_f
    KQ_r_reg = E_Regulation * KQ_r
    sumOdds = np.sum(KQ_f_reg) + np.sum(KQ_r_reg)

    kq_ge1_idx = np.where(KQ_f >= 1)
    kq_le1_idx = np.where(KQ_f < 1)
    kq_inv_ge1_idx = np.where(KQ_r > 1)
    kq_inv_le1_idx = np.where(KQ_r <= 1)
    
    epr = +np.sum(KQ_f_reg[kq_ge1_idx] * np.log(KQ_f[kq_ge1_idx]))/sumOdds \
          -np.sum(KQ_f_reg[kq_le1_idx] * np.log(KQ_f[kq_le1_idx]))/sumOdds \
          -np.sum(KQ_r_reg[kq_inv_le1_idx] * np.log(KQ_f[kq_inv_le1_idx]))/sumOdds \
          +np.sum(KQ_r_reg[kq_inv_ge1_idx] * np.log(KQ_f[kq_inv_ge1_idx]))/sumOdds
    return epr

def derivatives(log_vcounts,log_fcounts,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq, E_Regulation):

    nvar = log_vcounts.size
    log_metabolites = np.append(log_vcounts,log_fcounts) #log_counts
    EKQ_f = odds_alternate(E_Regulation,log_metabolites,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq, 1); #internal conversion to counts
    Keq_inverse = np.power(Keq,-1);
    EKQ_r = odds_alternate(E_Regulation,log_metabolites,mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs, Keq_inverse, -1);#internal conversion to counts
    
    s_mat = S_mat[:,0:nvar]
    deriv = s_mat.T.dot((EKQ_f - EKQ_r).T)
	
    return(deriv.reshape(deriv.size,))

def odds(log_counts,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, direction = 1):

    Q_inv = np.exp(-direction*(R_back_mat.dot(log_counts) + P_mat.dot(log_counts)))    
    KQ = np.multiply(Keq_constant,Q_inv)
    return(KQ)
    
def odds_alternate(E_Regulation,log_counts,mu0,S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq_constant, direction = 1):

    scale_min = np.min(-direction*(R_back_mat.dot(log_counts) + P_mat.dot(log_counts)))
    scale_max = np.max(-direction*(R_back_mat.dot(log_counts) + P_mat.dot(log_counts)))
    scale = (scale_max + scale_min)/2.0
    
    scaled_val = -direction*(R_back_mat.dot(log_counts) + P_mat.dot(log_counts)) - scale
    log_Q_inv = (-direction*(R_back_mat.dot(log_counts) + P_mat.dot(log_counts)))
    log_EKQ = np.log(np.multiply(E_Regulation,Keq_constant)) + log_Q_inv
    q_max = np.max(abs(log_Q_inv))
    ekq_max = np.max(abs(log_EKQ))
    
    if (q_max < ekq_max):
        Q_inv = np.exp(log_Q_inv)    
        KQ = np.multiply(Keq_constant,Q_inv)
        EKQ = np.multiply(E_Regulation,KQ)
    else:
        log_EKQ = np.log(np.multiply(E_Regulation,Keq_constant)) + log_Q_inv
        EKQ = np.exp(log_EKQ)
    return(EKQ)

def oddsDiff(log_vcounts, log_fcounts, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq, E_Regulation):
    
    log_metabolites = np.append(log_vcounts,log_fcounts)
    KQ_f = odds(log_metabolites, mu0, S_mat, R_back_mat, P_mat, delta_increment_for_small_concs, Keq);
    Keq_inverse = np.power(Keq,-1);
    KQ_r = odds(log_metabolites, mu0,-S_mat, P_mat, R_back_mat, delta_increment_for_small_concs,Keq_inverse,-1);
  
    #WARNING: Multiply regulation here, not on individual Keq values.
    KQdiff =  E_Regulation * (KQ_f - KQ_r);
    return(KQdiff)



def calc_Jac2(log_vcounts, log_fcounts, S_mat, delta_increment_for_small_concs, KQ_forward, KQ_reverse, E_Regulation):
#Jac is the Jacobian matrix, 
#an N metabolite time-differential equations by (rows) by 
#N metabolites derivatives (columns)
#J_ij = d/dx_i(df_j/dt)
    metabolites = np.append(np.exp(log_vcounts), np.exp(log_fcounts))
    delta_metabolites = metabolites + delta_increment_for_small_concs
    delta_recip_metabolites = np.power(delta_metabolites, -1)
    np.nan_to_num(delta_recip_metabolites, copy=False)

    s_ij_x_recip_metabolites = delta_recip_metabolites*(-S_mat)
    x = E_Regulation*(KQ_forward + KQ_reverse)
    y = ((x.T)*s_ij_x_recip_metabolites.T).T    
    RR = y * delta_metabolites.T
    Jac = np.matmul(S_mat.T,y)
    return (RR, Jac)


def calc_A(log_vcounts, log_fcounts, S_mat, Jac, E_Regulation):
# A is the linear stability matrix where the Aij = df_i/dx_j * x_j
# A is an N metabolite time-differential equations by (rows) by 
# N metabolites derivatives (columns)
#   J_ij = d/dx_i(df_j/dt)
# See Beard and Qian, Chemical Biophysics, p 156.
# row i: d/dc_i * dA/dt
# column j: d/dc dX_j/dt
# row i, column j: d/dc_i * dc_j/dt
    metabolites = np.append(np.exp(log_vcounts), np.exp(log_fcounts))
    A = (metabolites*Jac)
    return(A)

def conc_flux_control_coeff(nvar, A, S_mat, rxn_flux, RR):
    #ccc = d log(concentration) / d log (rate)
    #fcc = d log(flux) / d log(rate)
    
    #ccc = -B*S_mat*flux
    #fcc = delta_mn - 1/flux_m * (RBS)_mn * flux_n
    
    B = np.linalg.pinv(A[0:nvar,0:nvar]);
    ccc = np.matmul(-B, S_mat[:,0:nvar].T)*rxn_flux
    RB = np.matmul(RR[:,0:nvar], B)
    RBS = np.matmul(RB, S_mat[:, 0:nvar].T)
    
    rxn_flux_temp = rxn_flux;
    idx = np.where(rxn_flux_temp == 0.0)[0]
    rxn_flux_temp[idx] = np.finfo(float).tiny #avoid possible division by zero
    
    fcc_temp = (1.0/rxn_flux_temp) * (RBS * rxn_flux)    
    fcc = np.identity(len(fcc_temp)) - fcc_temp
    return [ccc,fcc]

def calc_deltaS(log_vcounts,target_log_vcounts, log_fcounts, S_mat, KQ):
    pt_forward=np.zeros(len(KQ))

    pt_reverse=np.zeros(len(KQ))

    log_target_metabolite = np.append(target_log_vcounts, log_fcounts)
    log_metabolite = np.append(log_vcounts, log_fcounts)

    delta_S_new = np.zeros(len(KQ))
    row, = np.where(KQ >= 1)
    P_Forward = (S_mat > 0)

    #necessary to loop over the rows instead 
    for rxn in row:
        forward_val = (np.multiply(P_Forward[rxn,:], log_metabolite))
        forward_target = (np.multiply(P_Forward[rxn,:], log_target_metabolite))
        pt_forward[rxn] = np.max(forward_val - forward_target)
        delta_S_new[rxn] = pt_forward[rxn]

    row, = np.where(KQ < 1)
    P_Reverse = (S_mat < 0)
    
    for rxn in row:
        reverse_val = (np.multiply(P_Reverse[rxn,:], log_metabolite))
        reverse_target = (np.multiply(P_Reverse[rxn,:], log_target_metabolite))
        pt_reverse[rxn] = np.max(reverse_val - reverse_target)
        delta_S_new[rxn] = pt_reverse[rxn]
    return delta_S_new

def calc_deltaS_metab(v_log_counts, target_v_log_counts ):
    delta_S_metab = v_log_counts - target_v_log_counts
    
    return delta_S_metab

def get_enzyme2regulate(ipolicy, delta_S_metab,delta_S, ccc, KQ, E_regulation, v_counts):
    
    reaction_choice=-1

    if (ipolicy == 'local') or (ipolicy == 1):
        
        sm_idx = [i for i,val in enumerate(delta_S_metab) if val > 0]
        S_index = [i for i,val in enumerate(delta_S) if val > 0]
        #print(S_index)
    else:
        sm_idx = [i for i,val in enumerate(delta_S_metab) if val > 0]
        S_index = [i for i,val in enumerate(delta_S)]
    
    if (len(S_index)>0 ):
        if (ipolicy == 'local') or (ipolicy == 1):
            
            temp = ccc[np.ix_(sm_idx,S_index)]#np.ix_ does outer product
                
            temp2 = (temp > 0) #ccc>0 means derivative is positive (dlog(conc)/dlog(activity)>0) 
            
            temp_x = (temp*temp2)#Do not use matmul, use element wise mult.

            #TEMPORARY TEST
            dx = np.multiply(v_counts[sm_idx].T,temp_x.T)
            
            #dx_neg = v_counts[sm_idx_neg].T*temp_x_neg
            DeltaAlpha = 0.001  # must be small enough such that the arguement
                                # of the log below is > 0
            DeltaDeltaS = -np.log(1 - DeltaAlpha * np.divide(dx,v_counts[sm_idx]))
            index3 = np.argmax(np.sum(DeltaDeltaS, axis=1)) #sum along rows (i.e. metabolites)
            reaction_choice = S_index[index3]

            
            return reaction_choice
        if (ipolicy == 'unrestricted') or (ipolicy == 2):
            temp = ccc[np.ix_(sm_idx,S_index)]#np.ix_ does outer product
                
            temp2 = (temp > 0) #ccc>0 means derivative is positive (dlog(conc)/dlog(activity)>0) 
            #this means regulation (decrease in activity) will result in decrease in conc
            
            temp_x = (temp*temp2)#Do not use matmul, use element wise mult.
            
            #rxn by metabolite matrix
            dx = np.multiply(v_counts[sm_idx].T,temp_x.T)
            
            DeltaAlpha = 0.001; # must be small enough such that the arguement
                                # of the log below is > 0
            DeltaDeltaS = -np.log(1 - DeltaAlpha*np.divide(dx,v_counts[sm_idx]))
            index3 = np.argmax(np.sum(DeltaDeltaS, axis=1)) #sum along rows (i.e. metabolites)
            reaction_choice = S_index[index3]
            
            return reaction_choice
    else:
        print("in function get_enzyme2regulate")
        print("all errors gone, fully uptimized")
        return -1

#input: current enzyme activities np.array(float) and reaction choice (int)
#output float
def calc_new_enzyme_simple(E_vec,React_Choice):
    current_E = E_vec[React_Choice]
    new_E = current_E - current_E/5.0
    return new_E

#use delta_S as args input variable to use method1 (E=E/2) when delta_S_val is small
def calc_reg_E_step(E_vec, React_Choice, nvar, log_vcounts, 
                    log_fcounts,complete_target_log_counts,S_mat, A, rxn_flux,KQ,
                    *args):
    
    varargin = args
    nargin = len(varargin)
    method = 1
    delta_S_val_method1=0.0
    
    vcounts = np.exp(log_vcounts)
    fcounts = np.exp(log_fcounts)
    E=E_vec[React_Choice]
    
        
    metabolite_counts = np.append(vcounts, fcounts)
    S_T=S_mat.T
    B=np.linalg.pinv(A[0:nvar,0:nvar])
    
    prod_indices=[]
    arr_temp = (S_mat[React_Choice,0:vcounts.size]) #set temporary to avoid 2d array in where function

    if (arr_temp.shape[0] == 1):
      #then arr_temp was a 2D array and we need to extract the 1D array inside.
      arr_temp = arr_temp[0]
      
    if(KQ[React_Choice] < 1):
        prod_indices = np.where( arr_temp < 0 )[0]    
    else:
        prod_indices = np.where( arr_temp > 0 )[0]
    E_choices=np.ones(len(prod_indices))
    newE1=1.0
    if (np.size(E_choices) == 0 ):
        newE = E
    else:
        for i in range(0,len(prod_indices)):
            prod_index = prod_indices[i]
            dx_j = metabolite_counts[prod_indices[i] ] - complete_target_log_counts[i]
            x_j_eq = metabolite_counts[ prod_indices[i] ];
            if (dx_j > delta_S_val_method1):
                delta_S_val_method1=dx_j
                
            TEMP=(S_T[0:len(vcounts),React_Choice])*(rxn_flux[React_Choice]) 
            TEMP2=np.matmul(-B[prod_index,:],  TEMP )
            deltaE = E * (dx_j/x_j_eq) * TEMP2
            E_choices[i] = deltaE;
            
        idx = np.argmax(E_choices)
        delta_E_Final = E_choices[idx]
        newE = E - E/5#(delta_E_Final)
        
        tolerance = 1.0e-07
        if ((newE < 0) or (newE > 1.0)):
            newE=E
        
        if (method == 1):
            if(delta_S_val_method1 > tolerance):
                newE = E - E/5
            else:
                newE = E - E/5#(delta_E_Final)         
                if(newE < 0) or (newE > 1):
                    newE = E - E/5
    return newE
