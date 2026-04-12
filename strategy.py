#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_Volume_Trend_v40
Hypothesis: Use daily Camarilla levels with 4h trend filter and volume confirmation on 4h timeframe.
Enter long when price breaks above daily H3 in uptrend (4h close > EMA34) with volume > 1.5x average.
Enter short when price breaks below daily L3 in downtrend (4h close < EMA34) with volume > 1.5x average.
Exit on trend reversal or price retracement to daily H4/L4 levels. Uses 0.30 position sizing.
Designed to capture strong directional moves in both bull and bear markets with low trade frequency.
Target: 50-150 total trades over 4 years (12-37/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_Volume_Trend_v40"
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
    
    # Calculate Daily Camarilla levels (H3, L3, H4, L4)
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
    
    # Align to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    h4_4h = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    l4_4h = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    
    # === 4H TREND FILTER ===
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === VOLUME SURGE FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or np.isnan(ema34[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Long: break above daily H3 in uptrend with volume surge
        long_signal = (close[i] > ema34[i] and 
                      close[i] > h3_4h[i] * 1.001 and  # Break above daily H3
                      vol_ratio[i] > 1.5)
        
        # Short: break below daily L3 in downtrend with volume surge
        short_signal = (close[i] < ema34[i] and 
                       close[i] < l3_4h[i] * 0.999 and  # Break below daily L3
                       vol_ratio[i] > 1.5)
        
        # Exit: trend reversal or retracement to daily H4/L4
        exit_long = (position == 1 and 
                    (close[i] <= ema34[i] or close[i] <= h4_4h[i]))
        exit_short = (position == -1 and 
                     (close[i] >= ema34[i] or close[i] >= l4_4h[i]))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.30
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.30 if position == 1 else (-0.30 if position == -1 else 0.0)
    
    return signals