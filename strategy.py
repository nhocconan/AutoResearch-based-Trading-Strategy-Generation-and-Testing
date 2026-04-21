#!/usr/bin/env python3
"""
12h_1d_Pivot_R1S1_Breakout_Volume_Trend
Hypothesis: Use 1d Camarilla R1/S1 breakouts with volume confirmation and 1d EMA50 trend filter on 12h timeframe.
Long when price breaks above R1 with volume > 1.5x 20-bar avg AND price > EMA50.
Short when price breaks below S1 with volume > 1.5x 20-bar avg AND price < EMA50.
Exit when price crosses back through the pivot point (PP).
Designed for 12h timeframe to capture multi-day moves with ~10-25 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
Volume and trend filters reduce false breakouts and whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot levels (based on previous day)
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp = np.full_like(close_1d, np.nan)
    r1 = np.full_like(close_1d, np.nan)
    s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(high_1d)):
        pp[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3.0
        r1[i] = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
        s1[i] = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12.0
    
    # Shift to align with current day (levels are based on previous day)
    pp = np.roll(pp, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    pp[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above R1 + volume confirmation + price above EMA50
            if price > r1_aligned[i] and volume_ok and price > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume confirmation + price below EMA50
            elif price < s1_aligned[i] and volume_ok and price < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0