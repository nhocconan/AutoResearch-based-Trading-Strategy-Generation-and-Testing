#!/usr/bin/env python3
"""
4h_1d_Camarilla_TrendBreakout_v2
Hypothesis: Breakout above H3 or below L3 with volume confirmation and daily trend filter.
Trades only when price breaks key daily levels with volume, using daily trend to avoid false breakouts.
Designed for low frequency: one condition per direction, volume filter, and trend alignment.
Works in bull via buying strength, in bear via selling weakness.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_TrendBreakout_v2"
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
    
    # === DAILY DATA ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily SMA50 for trend filter
    close_s = pd.Series(close_1d)
    sma50 = close_s.rolling(window=50, min_periods=50).mean().values
    
    # Previous day's close for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = close_1d[0]
    
    # Daily range
    range_1d = high_1d - low_1d
    
    # Camarilla levels based on previous day
    h3 = prev_close + (range_1d * 1.1 / 4)
    l3 = prev_close - (range_1d * 1.1 / 4)
    h4 = prev_close + (range_1d * 1.1)
    l4 = prev_close - (range_1d * 1.1)
    
    # Align to 4h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50)
    
    # Volume average (20-period for confirmation)
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        vol_avg[i] = vol_sum / vol_count if vol_count > 0 else 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Trend filter: price above/below SMA50
        price_above_sma = close[i] > sma50_aligned[i]
        price_below_sma = close[i] < sma50_aligned[i]
        
        # Breakout entries: long on H3 break in uptrend, short on L3 break in downtrend
        long_breakout = (close[i] > h3_aligned[i]) and price_above_sma and vol_confirm
        short_breakout = (close[i] < l3_aligned[i]) and price_below_sma and vol_confirm
        
        # Exit at opposite H3/L3 level
        exit_long = close[i] < l3_aligned[i]
        exit_short = close[i] > h3_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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