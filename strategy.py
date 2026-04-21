#!/usr/bin/env python3
"""
1h_4h_1d_Pivot_R1S1_Breakout_With_Volume
Hypothesis: Use 4h and 1d timeframe Pivot R1/S1 levels for trend direction, with 1h breakout entries.
In uptrend (price > 4h pivot AND price > 1d pivot), buy when price breaks above 4h R1 with volume confirmation.
In downtrend (price < 4h pivot AND price < 1d pivot), sell when price breaks below 4h S1 with volume confirmation.
Uses 4h ATR for volatility filter to avoid choppy markets.
Session filter (08-20 UTC) to avoid low-volume sessions.
Target: 15-37 trades/year on 1h timeframe (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_pivots(high, low, close):
    """Calculate Pivot Points: P, R1, R2, S1, S2"""
    pivot = (high + low + close) / 3.0
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    return pivot, r1, s1

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
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
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h and 1d data once for pivots and ATR
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h pivots and ATR
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    pivot_4h, r1_4h, s1_4h = calculate_pivots(high_4h, low_4h, close_4h)
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Calculate 1d pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d, _, _ = calculate_pivots(high_1d, low_1d, close_1d)
    
    # Align to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or 
            np.isnan(s1_4h_aligned[i]) or np.isnan(atr_4h_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
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
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        if i >= 20:
            vol_filter = atr_4h_aligned[i] > np.percentile(atr_4h_aligned[:i+1], 30)
        else:
            vol_filter = True
        
        if position == 0:
            # Uptrend: price > 4h pivot AND price > 1d pivot
            if price > pivot_4h_aligned[i] and price > pivot_1d_aligned[i]:
                # Long: price breaks above 4h R1 with volume confirmation
                if (price > r1_4h_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = 0.20
                    position = 1
            # Downtrend: price < 4h pivot AND price < 1d pivot
            elif price < pivot_4h_aligned[i] and price < pivot_1d_aligned[i]:
                # Short: price breaks below 4h S1 with volume confirmation
                if (price < s1_4h_aligned[i] and 
                    volume_ok and vol_filter):
                    signals[i] = -0.20
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal (price below either pivot) or volatility drops
            if price < pivot_4h_aligned[i] or price < pivot_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend reversal (price above either pivot) or volatility drops
            if price > pivot_4h_aligned[i] or price > pivot_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Pivot_R1S1_Breakout_With_Volume"
timeframe = "1h"
leverage = 1.0