#!/usr/bin/env python3
"""
4h_Daily_KeyLevels_Reversion_With_Volume
Hypothesis: Reversion to daily key levels (pivot, VWAP, round numbers) with volume confirmation and 4h trend filter. Works in bull markets by buying pullbacks to support, and in bear markets by selling rallies to resistance. Uses 1-day pivot points and VWAP for structure, 4h EMA for trend, and volume spike for confirmation. Targets 20-40 trades/year with tight entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivot(high, low, close):
    """Calculate standard pivot point and support/resistance levels"""
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    return pivot, r1, r2, s1, s2

def calculate_vwap(high, low, close, volume):
    """Calculate Volume Weighted Average Price"""
    typical = (high + low + close) / 3
    vwap = np.nancumsum(typical * volume) / np.nancumsum(volume)
    return vwap

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    ema = np.zeros_like(arr)
    alpha = 2 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for pivot points and VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = np.zeros(len(df_1d))
    r1_1d = np.zeros(len(df_1d))
    s1_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        pivot, r1, _, s1, _ = calculate_pivot(high_1d[i], low_1d[i], close_1d[i])
        pivot_1d[i] = pivot
        r1_1d[i] = r1
        s1_1d[i] = s1
    
    # Calculate daily VWAP
    vwap_1d = calculate_vwap(high_1d, low_1d, close_1d, df_1d['volume'].values)
    
    # Align daily levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Calculate 4h EMA for trend filter
    close_4h = prices['close'].values
    ema_4h = calculate_ema(close_4h, 21)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_4h[i]):
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
            # Long: price near S1 or VWAP with volume confirmation and above 4h EMA
            near_support = abs(price - s1_1d_aligned[i]) / price < 0.005 or abs(price - vwap_1d_aligned[i]) / price < 0.005
            if near_support and price > ema_4h[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price near R1 or VWAP with volume confirmation and below 4h EMA
            elif abs(price - r1_1d_aligned[i]) / price < 0.005 or abs(price - vwap_1d_aligned[i]) / price < 0.005:
                if price < ema_4h[i] and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches pivot or VWAP, or trend changes
            if price >= pivot_1d_aligned[i] or price >= vwap_1d_aligned[i] or price < ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches pivot or VWAP, or trend changes
            if price <= pivot_1d_aligned[i] or price <= vwap_1d_aligned[i] or price > ema_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Daily_KeyLevels_Reversion_With_Volume"
timeframe = "4h"
leverage = 1.0