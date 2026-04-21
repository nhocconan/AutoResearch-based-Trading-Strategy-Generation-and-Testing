#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime
Hypothesis: Buy when price breaks above Camarilla R1 on 1d with volume and chop regime filter; short when breaks below S1.
Uses 1d Camarilla levels and 12h ADX for regime filtering to avoid whipsaws.
Designed for 4h timeframe to capture 20-40 trades/year with high-probability entries.
Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    c = close
    r4 = c + range_val * 1.1 / 2
    r3 = c + range_val * 1.1 / 4
    r2 = c + range_val * 1.1 / 6
    r1 = c + range_val * 1.1 / 12
    s1 = c - range_val * 1.1 / 12
    s2 = c - range_val * 1.1 / 6
    s3 = c - range_val * 1.1 / 4
    s4 = c - range_val * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    
    for i in range(1, len(high)):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    plus_di = np.zeros_like(high)
    minus_di = np.zeros_like(high)
    if np.all(atr != 0):
        plus_di = 100 * plus_dm / atr
        minus_di = 100 * minus_dm / atr
    
    dx = np.zeros_like(high)
    if np.all((plus_di + minus_di) != 0):
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    
    adx = np.zeros_like(high)
    if len(dx) >= period:
        adx[period-1] = np.mean(dx[:period])
        for i in range(period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    r1_1d = np.zeros_like(close_1d)
    r2_1d = np.zeros_like(close_1d)
    s1_1d = np.zeros_like(close_1d)
    s2_1d = np.zeros_like(close_1d)
    
    for i in range(len(close_1d)):
        r1, r2, _, _, s1, s2, _, _ = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        r1_1d[i] = r1
        r2_1d[i] = r2
        s1_1d[i] = s1
        s2_1d[i] = s2
    
    # Load 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Align Camarilla levels to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(adx_12h_aligned[i])):
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
        
        # Regime filter: ADX > 25 for trending market
        regime_ok = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above R1 with volume and regime
            if (price > r1_1d_aligned[i] and 
                volume_ok and regime_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and regime
            elif (price < s1_1d_aligned[i] and 
                  volume_ok and regime_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or regime changes
            if (price < s1_1d_aligned[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or regime changes
            if (price > r1_1d_aligned[i] or 
                adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0