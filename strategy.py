#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Breakout_Trend
Hypothesis: Use weekly and daily Camarilla levels with 1d trend filter and volume confirmation on 12h timeframe.
Enter long when price breaks above weekly H3 in uptrend (1d close > EMA20) with volume > 1.5x average.
Enter short when price breaks below weekly L3 in downtrend (1d close < EMA20) with volume > 1.5x average.
Exit on trend reversal or price retracement to weekly H4/L4 levels. Uses 0.25 position sizing.
Designed to capture strong directional moves in both bull and bear markets with low trade frequency.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Breakout_Trend"
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
    
    # === WEEKLY CAMARILLA LEVELS ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Weekly Camarilla levels (H3, L3, H4, L4)
    camarilla_h3_1w = np.full(len(close_1w), np.nan)
    camarilla_l3_1w = np.full(len(close_1w), np.nan)
    camarilla_h4_1w = np.full(len(close_1w), np.nan)
    camarilla_l4_1w = np.full(len(close_1w), np.nan)
    
    for i in range(len(close_1w)):
        if np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i]):
            continue
        range_val = high_1w[i] - low_1w[i]
        camarilla_h3_1w[i] = close_1w[i] + range_val * 1.1 / 6
        camarilla_l3_1w[i] = close_1w[i] - range_val * 1.1 / 6
        camarilla_h4_1w[i] = close_1w[i] + range_val * 1.1 / 4
        camarilla_l4_1w[i] = close_1w[i] - range_val * 1.1 / 4
    
    # Align to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    l3_12h = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    h4_12h = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    l4_12h = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    
    # === DAILY TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_12h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(ema20_1d_12h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current 1d close 
        close_1d_arr = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
        trend_up = close_1d_aligned[i] > ema20_1d_12h[i]
        trend_down = close_1d_aligned[i] < ema20_1d_12h[i]
        
        # Long: break above weekly H3 in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] > h3_12h[i] * 1.001 and  # Break above weekly H3
                      vol_ratio[i] > 1.5)
        
        # Short: break below weekly L3 in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] < l3_12h[i] * 0.999 and  # Break below weekly L3
                       vol_ratio[i] > 1.5)
        
        # Exit: trend reversal or retracement to weekly H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] <= h4_12h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] >= l4_12h[i]))
        
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