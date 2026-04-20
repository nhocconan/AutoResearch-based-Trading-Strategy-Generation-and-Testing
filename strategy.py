#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume
Hypothesis: Use daily Camarilla pivot levels (R1, S1) as support/resistance. 
Break above R1 with volume confirmation = long; break below S1 with volume confirmation = short.
Filter trades using 12h ADX > 25 to ensure trending markets (avoid whipsaw in ranging markets).
Position size 0.25 to balance risk and reward. Target: 15-35 trades/year per symbol.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue).
"""

name = "12h_1d_Camarilla_Pivot_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 12h timeframe (wait for daily close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 12h ADX(14) for trend filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Plus Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Wilder's smoothing
        def wilder_smoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            alpha = 1.0 / period
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smoothing(tr, period)
        # Avoid division by zero
        atr_safe = np.where(atr == 0, 1e-10, atr)
        plus_di = 100 * wilder_smoothing(plus_dm, period) / atr_safe
        minus_di = 100 * wilder_smoothing(minus_dm, period) / atr_safe
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilder_smoothing(dx, period)
        return adx
    
    adx_12h = calculate_adx(high, low, close, 14)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = np.zeros_like(volume)
    vol_ma[:] = np.nan
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(adx_12h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume and ADX > 25
            if (close[i] > r1_1d_aligned[i] and 
                vol_filter[i] and 
                adx_12h[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and ADX > 25
            elif (close[i] < s1_1d_aligned[i] and 
                  vol_filter[i] and 
                  adx_12h[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot OR ADX weak (< 20)
            if close[i] < pivot_1d_aligned[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot OR ADX weak (< 20)
            if close[i] > pivot_1d_aligned[i] or adx_12h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals