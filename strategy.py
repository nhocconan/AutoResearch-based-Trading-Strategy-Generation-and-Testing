#!/usr/bin/env python3
"""
4h_12h_Camarilla_3level_Trend_Filter
Hypothesis: On 4h timeframe, take long positions when price touches Camarilla L3 support in uptrend,
and short positions when price touches Camarilla H3 resistance in downtrend.
Use 12h EMA21 for trend filter and volume > 1.3x average for confirmation.
Exit when price reaches opposite H3/L3 level or trend reverses.
Designed for mean-reversion within trend with low trade frequency.
Target: 50-150 total trades over 4 years (12-37/year) on 4h timeframe.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_3level_Trend_Filter"
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
    
    # === 12H DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # === DAILY DATA FOR CAMARILLA LEVELS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_h3 = []
    camarilla_l3 = []
    for i in range(len(df_1d)):
        if i == 0:
            # For first day, use same values (will be filtered by alignment delay)
            camarilla_h3.append(high_1d[i])
            camarilla_l3.append(low_1d[i])
        else:
            # Camarilla H3/L3 from previous day
            range_prev = high_1d[i-1] - low_1d[i-1]
            close_prev = close_1d[i-1]
            h3 = close_prev + range_prev * 1.1 / 6
            l3 = close_prev - range_prev * 1.1 / 6
            camarilla_h3.append(h3)
            camarilla_l3.append(l3)
    
    camarilla_h3 = np.array(camarilla_h3)
    camarilla_l3 = np.array(camarilla_l3)
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema21_12h_aligned[i]) or np.isnan(h3_4h[i]) or 
            np.isnan(l3_4h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from 12h EMA21
        close_12h_arr = df_12h['close'].values
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h_arr)
        trend_up = close_12h_aligned[i] > ema21_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema21_12h_aligned[i]
        
        # Entry conditions: price near Camarilla level + trend + volume
        near_h3 = abs(high[i] - h3_4h[i]) / h3_4h[i] < 0.002  # Within 0.2% of H3
        near_l3 = abs(low[i] - l3_4h[i]) / l3_4h[i] < 0.002   # Within 0.2% of L3
        
        long_signal = near_l3 and trend_up and vol_ratio[i] > 1.3
        short_signal = near_h3 and trend_down and vol_ratio[i] > 1.3
        
        # Exit conditions: reach opposite level or trend reversal
        exit_long = (position == 1 and 
                    (near_h3 or not trend_up))
        exit_short = (position == -1 and 
                     (near_l3 or not trend_down))
        
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