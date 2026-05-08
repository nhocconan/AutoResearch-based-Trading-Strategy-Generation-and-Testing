#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Supertrend trend filter with Camarilla pivot reversals at R1/S1 levels
# Uses 1d Camarilla levels for reversal entries in direction of 1d Supertrend.
# Volume confirmation ensures momentum behind reversals.
# Designed for fewer trades (target 25-40/year) with clear reversal logic in both bull/bear markets.

name = "4h_Camarilla_Rev_1dSupertrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla and Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Supertrend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    atr_period = 10
    atr_mult = 3.0
    
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.zeros_like(close_1d)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    basic_ub = (high_1d + low_1d) / 2 + atr_mult * atr
    basic_lb = (high_1d + low_1d) / 2 - atr_mult * atr
    
    final_ub = np.zeros_like(close_1d)
    final_lb = np.zeros_like(close_1d)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(close_1d)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    supertrend = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            supertrend[i] = final_ub[i]
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close_1d[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
            else:
                if close_1d[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                else:
                    supertrend[i] = final_ub[i]
    
    # Calculate 1d Camarilla levels (using previous day's data)
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    camarilla_r1[0] = camarilla_s1[0] = close_1d[0]  # placeholder for first day
    
    for i in range(1, len(close_1d)):
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        camarilla_r1[i] = close_prev + 1.1 * (high_prev - low_prev) / 12
        camarilla_s1[i] = close_prev - 1.1 * (high_prev - low_prev) / 12
    
    # Align indicators to 4h
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(supertrend_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_avg_today = vol_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirmed = vol_current > 1.5 * vol_avg_today
        
        if position == 0:
            # Look for reversals at Camarilla levels in direction of Supertrend
            if supertrend_aligned[i] > 0:  # Uptrend - look for longs at S1
                if price <= camarilla_s1_aligned[i] * 1.001 and price >= camarilla_s1_aligned[i] * 0.999:
                    if vol_confirmed:
                        signals[i] = 0.25
                        position = 1
                        continue
            else:  # Downtrend - look for shorts at R1
                if price >= camarilla_r1_aligned[i] * 0.999 and price <= camarilla_r1_aligned[i] * 1.001:
                    if vol_confirmed:
                        signals[i] = -0.25
                        position = -1
                        continue
        
        elif position == 1:
            # Exit long: price crosses Supertrend or reaches R1
            if supertrend_aligned[i] <= 0 or price >= camarilla_r1_aligned[i] * 0.999:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses Supertrend or reaches S1
            if supertrend_aligned[i] >= 0 or price <= camarilla_s1_aligned[i] * 1.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals