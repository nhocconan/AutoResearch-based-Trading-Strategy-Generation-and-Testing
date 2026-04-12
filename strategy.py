#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_v1
Hypothesis: Use weekly market regime (bull/bear/range) with daily Camarilla H3/L3 breakouts for trend-following entries.
In weekly uptrend (weekly close > weekly SMA20): go long on break above daily H3 with volume > 1.5x average.
In weekly downtrend (weekly close < weekly SMA20): go short on break below daily L3 with volume > 1.5x average.
In weekly range (otherwise): no new entries, only exit existing positions.
Exit when price returns to daily H4/L4 levels or weekly trend reverses.
Position size: 0.25 to limit drawdown. Designed for 1d timeframe to capture multi-day moves with low frequency.
Target: 20-60 total trades over 4 years (5-15/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_v1"
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
    
    # === WEEKLY TREND (REGIME) FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    weekly_close = df_1w['close'].values
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    weekly_uptrend = weekly_close_aligned > sma20_1w_aligned
    weekly_downtrend = weekly_close_aligned < sma20_1w_aligned
    
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
    
    h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_1d = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or np.isnan(h4_1d[i]) or 
            np.isnan(l4_1d[i]) or np.isnan(weekly_uptrend[i]) or np.isnan(weekly_downtrend[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: weekly uptrend + break above daily H3 + volume surge
        long_signal = (weekly_uptrend[i] and 
                      close[i] > h3_1d[i] * 1.001 and  # Break above H3
                      vol_ratio[i] > 1.5)
        
        # Short: weekly downtrend + break below daily L3 + volume surge
        short_signal = (weekly_downtrend[i] and 
                       close[i] < l3_1d[i] * 0.999 and  # Break below L3
                       vol_ratio[i] > 1.5)
        
        # Exit: price returns to H4/L4 or weekly trend reverses
        exit_long = (position == 1 and 
                    (close[i] <= h4_1d[i] or not weekly_uptrend[i]))
        exit_short = (position == -1 and 
                     (close[i] >= l4_1d[i] or not weekly_downtrend[i]))
        
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