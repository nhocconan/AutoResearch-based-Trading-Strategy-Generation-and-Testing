#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v2
Hypothesis: Breakout of Camarilla R1/S1 levels on 12h with 1d volume spike and 1w trend filter.
Works in bull/bear by following weekly trend direction (EMA34) and using mean-reversion pivot breakouts.
Target: 12-30 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels (R1, S1, R4, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R4, S4
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    r4 = prev_close + 1.1 * rang / 2
    s4 = prev_close - 1.1 * rang / 2
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 1d data for volume filter (volume spike)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = vol_1d_aligned[i]  # already aligned 1d volume
        vol_ma = vol_ma_1d_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        volume_ok = volume > 2.0 * vol_ma
        
        # Trend filter: price above/below weekly EMA34
        uptrend = price > ema_34_1w_aligned[i]
        downtrend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long conditions: breakout above R1 with volume spike and uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: breakdown below S1 with volume spike and downtrend
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach R4 or reverse below R1
            if price >= r4_aligned[i] or price <= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach S4 or reverse above S1
            if price <= s4_aligned[i] or price >= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_v2"
timeframe = "12h"
leverage = 1.0