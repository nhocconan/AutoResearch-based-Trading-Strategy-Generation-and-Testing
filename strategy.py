#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation
Hypothesis: 12h timeframe with daily Camarilla pivot breakouts and volume confirmation.
Uses daily R1/S1 levels for structure and 12h price action for entry timing.
Designed to capture multi-day breakouts in trending markets while filtering out noise.
Volume confirmation ensures institutional participation. Targets 12-37 trades/year.
Works in bull markets (breakouts above R1) and bear markets (breakdowns below S1).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels"""
    typical = (high + low + close) / 3
    range_val = high - low
    R4 = close + range_val * 1.1 / 2
    R3 = close + range_val * 1.1 / 4
    R2 = close + range_val * 1.1 / 6
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    S2 = close - range_val * 1.1 / 6
    S3 = close - range_val * 1.1 / 4
    S4 = close - range_val * 1.1 / 2
    return R1, R2, R3, R4, S1, S2, S3, S4

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
    
    # Load 1d data once for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    R1_1d = np.zeros(len(df_1d))
    S1_1d = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        R1, _, _, _, S1, _, _, _ = calculate_camarilla_pivot(high_1d[i], low_1d[i], close_1d[i])
        R1_1d[i] = R1
        S1_1d[i] = S1
    
    # Align Camarilla levels to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Calculate 1-day ATR for volatility filter
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]):
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
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d_aligned[i] > np.percentile(atr_1d_aligned[:i+1], 30) if i >= 30 else True
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation
            if price > R1_1d_aligned[i] and volume_ok and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation
            elif price < S1_1d_aligned[i] and volume_ok and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or volatility drops
            if price < S1_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or volatility drops
            if price > R1_1d_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0