#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R1S1_Breakout_Volume_Control_v1
Hypothesis: Fade at Camarilla R1/S1 with volume confirmation during 08-20 UTC, using 1-week trend filter (price above/below weekly EMA20) to align with higher timeframe momentum. Works in bull/bear by taking mean-reversion trades at R1/S1 only when weekly trend supports the move. Targets 15-30 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    typical = (high + low + close) / 3
    range_val = high - low
    
    # Camarilla levels
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend direction
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, 20)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d = np.zeros(len(close_1d))
    S1_1d = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        R1, _, _, _, S1, _, _, _ = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(ema_1w_aligned[i]) or np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]):
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
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter
        weekly_uptrend = price > ema_1w_aligned[i]
        weekly_downtrend = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long: price crosses above S1 with volume, in weekly uptrend
            if (i >= 1 and 
                price > S1_1d_aligned[i] and 
                prices['close'].iloc[i-1] <= S1_1d_aligned[i-1] and
                weekly_uptrend and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume, in weekly downtrend
            elif (i >= 1 and 
                  price < R1_1d_aligned[i] and 
                  prices['close'].iloc[i-1] >= R1_1d_aligned[i-1] and
                  weekly_downtrend and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 or weekly trend turns down
            if price >= R1_1d_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 or weekly trend turns up
            if price <= S1_1d_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_Pivot_R1S1_Breakout_Volume_Control_v1"
timeframe = "6h"
leverage = 1.0