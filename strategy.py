#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Trend_Filter_v1
Hypothesis: Use weekly price position (above/below weekly midpoint) as trend filter for 12h chart.
Enter long when price breaks above daily H3 in weekly uptrend with volume confirmation.
Enter short when price breaks below daily L3 in weekly downtrend with volume confirmation.
Exit on trend reversal or price retracement to H4/L4 levels.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly midpoint: (weekly high + weekly low) / 2
    weekly_midpoint = (high_1w + low_1w) / 2
    weekly_uptrend = close_1w > weekly_midpoint
    weekly_downtrend = close_1w < weekly_midpoint
    
    weekly_uptrend_12h = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_12h = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3, H4, L4)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(weekly_uptrend_12h[i]) or np.isnan(weekly_downtrend_12h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: break above H3 in weekly uptrend with volume surge
        long_signal = (weekly_uptrend_12h[i] > 0.5 and 
                      close[i] > h3_12h[i] * 1.001 and  # Break above H3
                      vol_ratio[i] > 1.5)
        
        # Short: break below L3 in weekly downtrend with volume surge
        short_signal = (weekly_downtrend_12h[i] > 0.5 and 
                       close[i] < l3_12h[i] * 0.999 and  # Break below L3
                       vol_ratio[i] > 1.5)
        
        # Exit: trend reversal or retracement to H4/L4
        exit_long = (position == 1 and 
                    (weekly_uptrend_12h[i] < 0.5 or close[i] <= h4_12h[i]))
        exit_short = (position == -1 and 
                     (weekly_downtrend_12h[i] < 0.5 or close[i] >= l4_12h[i]))
        
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