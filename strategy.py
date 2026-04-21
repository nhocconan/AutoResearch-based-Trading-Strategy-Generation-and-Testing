#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeAndTrendFilter
Hypothesis: Camarilla pivot breakouts on 4h with volume confirmation and 1d trend filter (EMA34). Works in bull/bear by only taking long when price > 1d EMA34, short when price < 1d EMA34. Uses volume spike (>1.5x 20-bar average) to avoid false breakouts. Targets 20-40 trades/year with tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    range_val = high - low
    if range_val <= 0:
        return close, close, close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r4, r3, r2, r1, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels for each 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    r4 = np.full(len(df_1d), np.nan)
    r3 = np.full(len(df_1d), np.nan)
    r2 = np.full(len(df_1d), np.nan)
    r1 = np.full(len(df_1d), np.nan)
    s1 = np.full(len(df_1d), np.nan)
    s2 = np.full(len(df_1d), np.nan)
    s3 = np.full(len(df_1d), np.nan)
    s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        r4[i], r3[i], r2[i], r1[i], s1[i], s2[i], s3[i], s4[i] = calculate_camarilla(
            high_1d[i], low_1d[i], close_1d_arr[i]
        )
    
    # Align Camarilla levels to 4h
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
            # Long: price breaks above R1 + price > 1d EMA34 (uptrend) + volume
            if price > r1_aligned[i] and price > ema_1d_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + price < 1d EMA34 (downtrend) + volume
            elif price < s1_aligned[i] and price < ema_1d_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or trend change (price < EMA34)
            if price < s1_aligned[i] or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or trend change (price > EMA34)
            if price > r1_aligned[i] or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeAndTrendFilter"
timeframe = "4h"
leverage = 1.0