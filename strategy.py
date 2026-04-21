#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_R1S1_Breakout_With_4h_Trend_Filter
Hypothesis: 1h breakout strategy using daily Camarilla pivots for structure and 4h trend for direction.
Long when price breaks above R1 with 4h uptrend (price > 4h EMA20), short when price breaks below S1 with 4h downtrend (price < 4h EMA20).
Volume confirmation reduces false breakouts. Session filter (08-20 UTC) avoids low-volume periods.
Designed for 1h timeframe to target 15-37 trades/year with tight entry conditions.
4h trend filter prevents counter-trend trades in choppy markets, improving win rate in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    typical = (high + low + close) / 3
    range_val = high - low
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    return R1, S1

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d = np.zeros(len(df_1d))
    S1_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        R1, S1 = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    # Align Camarilla levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend
    close_4h = df_4h['close'].values
    ema_20_4h = calculate_ema(close_4h, 20)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(ema_20_4h_aligned[i]):
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
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume
            if price > R1_1d_aligned[i] and price > ema_20_4h_aligned[i] and volume_ok:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume
            elif price < S1_1d_aligned[i] and price < ema_20_4h_aligned[i] and volume_ok:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or 4h trend turns down
            if price < S1_1d_aligned[i] or price < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price breaks above R1 or 4h trend turns up
            if price > R1_1d_aligned[i] or price > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_Pivot_R1S1_Breakout_With_4h_Trend_Filter"
timeframe = "1h"
leverage = 1.0