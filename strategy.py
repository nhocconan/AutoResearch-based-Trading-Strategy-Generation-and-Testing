#!/usr/bin/env python3
"""
4h_12h_1d_Camarilla_Breakout_v3
Hypothesis: Breakout above daily Camarilla H3 or below L3 with 12h trend filter and volume confirmation.
Long when price breaks above H3 in uptrend (12h close > EMA20) with volume surge; short when breaks below L3 in downtrend.
Exit on trend reversal or retracement to H4/L4. Designed for 4h timeframe to capture directional moves in both bull and bear markets.
Target: 80-160 total trades over 4 years (20-40/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_Camarilla_Breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
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
    
    # Align to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 12-HOUR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_4h = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(ema20_12h_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Get current 12h close aligned to 4h
        close_12h_arr = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
        trend_up = close_12h_aligned[i] > ema20_12h_4h[i]
        trend_down = close_12h_aligned[i] < ema20_12h_4h[i]
        
        # Long: break above H3 in uptrend with volume surge
        long_signal = (trend_up and 
                      close[i] > h3_4h[i] * 1.001 and  # Break above H3
                      vol_ratio[i] > 2.0)
        
        # Short: break below L3 in downtrend with volume surge
        short_signal = (trend_down and 
                       close[i] < l3_4h[i] * 0.999 and  # Break below L3
                       vol_ratio[i] > 2.0)
        
        # Exit: trend reversal or retracement to H4/L4
        exit_long = (position == 1 and 
                    (not trend_up or close[i] <= h4_4h[i]))
        exit_short = (position == -1 and 
                     (not trend_down or close[i] >= l4_4h[i]))
        
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