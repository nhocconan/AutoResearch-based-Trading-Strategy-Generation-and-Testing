#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation
Hypothesis: Trade Camarilla pivot level (R1/S1) breakouts on 4h timeframe with 1d trend filter and volume confirmation. Works in bull/bear by only taking long when price > 1d EMA50 (uptrend) and short when price < 1d EMA50 (downtrend). Uses volume spike confirmation to avoid false breakouts. Targets 20-30 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    typical = (high + low + close) / 3
    range_ = high - low
    r1 = close + range_ * 1.1 / 12
    s1 = close - range_ * 1.1 / 12
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.zeros(len(df_1d))
    camarilla_s1 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
    
    # Calculate 1d EMA50 for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d indicators to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema_50_aligned[i]):
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
            # Long: price breaks above R1 + price > 1d EMA50 (uptrend) + volume
            if price > camarilla_r1_aligned[i] and price > ema_50_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + price < 1d EMA50 (downtrend) + volume
            elif price < camarilla_s1_aligned[i] and price < ema_50_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or price < 1d EMA50 (trend change)
            if price < camarilla_s1_aligned[i] or price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or price > 1d EMA50 (trend change)
            if price > camarilla_r1_aligned[i] or price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0