#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_pivot_volume_filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels from weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_h4 = []  # Resistance level 4 (strong resistance)
    camarilla_l4 = []  # Support level 4 (strong support)
    camarilla_h3 = []  # Resistance level 3
    camarilla_l3 = []  # Support level 3
    
    for i in range(len(close_1w)):
        high_val = high_1w[i]
        low_val = low_1w[i]
        close_val = close_1w[i]
        range_val = high_val - low_val
        
        # Camarilla formulas
        h4 = close_val + range_val * 1.5000
        l4 = close_val - range_val * 1.5000
        h3 = close_val + range_val * 1.2500
        l3 = close_val - range_val * 1.2500
        
        camarilla_h4.append(h4)
        camarilla_l4.append(l4)
        camarilla_h3.append(h3)
        camarilla_l3.append(l3)
    
    camarilla_h4 = np.array(camarilla_h4)
    camarilla_l4 = np.array(camarilla_l4)
    camarilla_h3 = np.array(camarilla_h3)
    camarilla_l3 = np.array(camarilla_l3)
    
    # Align to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Volume filter: current volume > 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price touching Camarilla levels with volume confirmation
        # Long when price touches L3/L4 support with volume
        long_signal = (close[i] <= l3_aligned[i] * 1.001 or close[i] <= l4_aligned[i] * 1.001) and volume_filter[i]
        # Short when price touches H3/H4 resistance with volume
        short_signal = (close[i] >= h3_aligned[i] * 0.999 or close[i] >= h4_aligned[i] * 0.999) and volume_filter[i]
        
        # Exit when price moves back toward center (mean reversion completion)
        pivot_center = (h3_aligned[i] + l3_aligned[i]) / 2
        exit_long = close[i] >= pivot_center * 0.999  # Price moved back up from support
        exit_short = close[i] <= pivot_center * 1.001  # Price moved back down from resistance
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals