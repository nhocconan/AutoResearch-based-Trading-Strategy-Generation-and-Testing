#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight
Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 12h EMA21 trend filter. Works in bull/bear by taking long at R1 breakout when price > 12h EMA21, short at S1 breakout when price < 12h EMA21. Uses volume > 1.3x 20-period average. Targets 20-50 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    return r1, s1

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for EMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA21 for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, 21)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    r1_1d, s1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        vol_ma = prices['volume'].iloc[i-20:i].mean()
        volume_ok = volume > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above R1 + price > 12h EMA21 (uptrend) + volume
            if price > r1_1d_aligned[i] and price > ema_12h_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + price < 12h EMA21 (downtrend) + volume
            elif price < s1_1d_aligned[i] and price < ema_12h_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S1 or price < 12h EMA21 (trend change)
            if price < s1_1d_aligned[i] or price < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R1 or price > 12h EMA21 (trend change)
            if price > r1_1d_aligned[i] or price > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeFilter_Tight"
timeframe = "4h"
leverage = 1.0