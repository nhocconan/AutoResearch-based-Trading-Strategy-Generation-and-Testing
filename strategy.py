#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_v3
Hypothesis: Use daily Camarilla H3/L3 breakouts with 4h EMA trend filter and volume spike confirmation.
Enter long when price breaks above H3 in uptrend (4h close > EMA20) with volume > 2x average.
Enter short when price breaks below L3 in downtrend (4h close < EMA20) with volume > 2x average.
Exit on trend reversal or price retracement to H4/L4 levels. Uses 0.25 position sizing to limit drawdown.
Designed to capture strong directional moves in both bull and bear markets while avoiding whipsaws.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_v3"
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
    
    # === 4-HOUR TREND FILTER ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    # EMA20 already aligned since we're on 4h timeframe
    ema20_4h_4h = ema20_4h
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(ema20_4h_4h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend conditions (already on 4h)
        trend_up = close_4h[i] > ema20_4h_4h[i]
        trend_down = close_4h[i] < ema20_4h_4h[i]
        
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