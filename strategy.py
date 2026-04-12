#!/usr/bin/env python3
"""
12h_1d_Camarilla_Volume_Momentum
Hypothesis: Use daily Camarilla H3/L3 levels with 12h momentum and volume confirmation to capture mean-reversion bounces in uptrends and breakdowns in downtrends. Works in bull markets (buying dips) and bear markets (selling rallies) by aligning with 12h trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Volume_Momentum"
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
    
    # === DAILY CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
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
    
    # === 12H MOMENTUM (Price > Open) ===
    open_12h = df_1d['open'].values  # Use daily open as proxy for 12h session open
    open_12h_aligned = align_htf_to_ltf(prices, df_1d, open_12h)
    bullish_momentum = close > open_12h_aligned
    bearish_momentum = close < open_12h_aligned
    
    # === VOLUME SURGE ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(open_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price touches L3 with bullish momentum and volume surge
        long_signal = (bullish_momentum[i] and 
                      close[i] <= l3_12h[i] * 1.005 and  # Slight tolerance for wicks
                      close[i] >= l4_12h[i] * 0.995 and  # Above L4
                      vol_ratio[i] > 1.8)
        
        # Short: price touches H3 with bearish momentum and volume surge
        short_signal = (bearish_momentum[i] and 
                       close[i] >= h3_12h[i] * 0.995 and  # Slight tolerance for wicks
                       close[i] <= h4_12h[i] * 1.005 and  # Below H4
                       vol_ratio[i] > 1.8)
        
        # Exit: momentum reversal or touch of opposite level
        exit_long = (position == 1 and 
                    (not bullish_momentum[i] or close[i] >= h3_12h[i]))
        exit_short = (position == -1 and 
                     (not bearish_momentum[i] or close[i] <= l3_12h[i]))
        
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