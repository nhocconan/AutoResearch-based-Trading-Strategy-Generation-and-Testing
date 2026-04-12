#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Breakout_Volume
Hypothesis: Use weekly (1w) and daily (1d) Camarilla levels as strong support/resistance zones.
Enter long when price breaks above daily H3 in a weekly uptrend (price > weekly H3) with volume > 2x average.
Enter short when price breaks below daily L3 in a weekly downtrend (price < weekly L3) with volume > 2x average.
Exit on trend reversal or price retracement to daily H4/L4 levels.
Designed for 12h timeframe to reduce trade frequency and capture major trend moves in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_Breakout_Volume"
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
    
    # === WEEKLY TREND CONTEXT (1w) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (H3, L3)
    camarilla_h3_1w = np.full(len(close_1w), np.nan)
    camarilla_l3_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        range_val = high_1w[i] - low_1w[i]
        camarilla_h3_1w[i] = close_1w[i] + range_val * 1.1 / 6
        camarilla_l3_1w[i] = close_1w[i] - range_val * 1.1 / 6
    
    # Align weekly levels to 12h timeframe
    h3_1w_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    l3_1w_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    
    # === DAILY CAMARILLA LEVELS (1d) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3, L3, H4, L4)
    camarilla_h3_1d = np.full(len(close_1d), np.nan)
    camarilla_l3_1d = np.full(len(close_1d), np.nan)
    camarilla_h4_1d = np.full(len(close_1d), np.nan)
    camarilla_l4_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]):
            continue
        range_val = high_1d[i] - low_1d[i]
        camarilla_h3_1d[i] = close_1d[i] + range_val * 1.1 / 6
        camarilla_l3_1d[i] = close_1d[i] - range_val * 1.1 / 6
        camarilla_h4_1d[i] = close_1d[i] + range_val * 1.1 / 4
        camarilla_l4_1d[i] = close_1d[i] - range_val * 1.1 / 4
    
    # Align daily levels to 12h timeframe
    h3_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    h4_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    l4_1d_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_1w_12h[i]) or np.isnan(l3_1w_12h[i]) or 
            np.isnan(h3_1d_12h[i]) or np.isnan(l3_1d_12h[i]) or
            np.isnan(h4_1d_12h[i]) or np.isnan(l4_1d_12h[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend: price relative to weekly H3/L3
        weekly_uptrend = close[i] > h3_1w_12h[i]
        weekly_downtrend = close[i] < l3_1w_12h[i]
        
        # Long: break above daily H3 in weekly uptrend with volume surge
        long_signal = (weekly_uptrend and 
                      close[i] > h3_1d_12h[i] * 1.001 and  # Break above daily H3
                      vol_ratio[i] > 2.0)
        
        # Short: break below daily L3 in weekly downtrend with volume surge
        short_signal = (weekly_downtrend and 
                       close[i] < l3_1d_12h[i] * 0.999 and  # Break below daily L3
                       vol_ratio[i] > 2.0)
        
        # Exit: weekly trend reversal or retracement to daily H4/L4
        exit_long = (position == 1 and 
                    (not weekly_uptrend or close[i] <= h4_1d_12h[i]))
        exit_short = (position == -1 and 
                     (not weekly_downtrend or close[i] >= l4_1d_12h[i]))
        
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