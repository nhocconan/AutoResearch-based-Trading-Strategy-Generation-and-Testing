#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend
Hypothesis: Use weekly CLOSE (not daily) to define trend direction, and daily Camarilla H3/L3 breakouts for entries.
This avoids whipsaws by requiring higher-timeframe trend confirmation. Exit on trend reversal or price retracement to H4/L4.
Target: 20-50 total trades over 4 years (5-12/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend"
timeframe = "1d"
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
    
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
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
    
    # Align to 1d timeframe
    h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_1d = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        trend_up = close[i] > sma50_1w_aligned[i]
        trend_down = close[i] < sma50_1w_aligned[i]
        
        # Long: break above H3 in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] > h3_1d[i] and 
                      vol_ratio[i] > 2.0)
        
        # Short: break below L3 in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] < l3_1d[i] and 
                       vol_ratio[i] > 2.0)
        
        # Exit: trend reversal or retracement to H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] <= h4_1d[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] >= l4_1d[i]))
        
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