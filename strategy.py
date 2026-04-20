#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and ADX filter.
# Uses daily Camarilla pivot levels from previous day for institutional reference.
# Entry requires breakout of R1 (long) or S1 (short) with volume > 1.5x 20-period average.
# ADX(14) > 25 ensures trending market to avoid chop. Position size 0.25.
# Works in bull/bear by following intraday momentum with institutional levels.
# Target: 20-40 trades per year to minimize fee drag.

name = "4h_Camarilla_R1S1_Volume_ADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Previous day's Camarilla pivot levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Calculate R1 and S1
    r1 = pivot + 1.1 * (high_1d - low_1d) / 12.0
    s1 = pivot - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align to 4h (use previous day's levels for current day's trading)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Volume spike confirmation ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d > 0, vol_ma_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h ADX for trend filter ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _wilder_smoothing(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    period = 14
    tr_sum = _wilder_smoothing(tr, period)
    plus_dm_sum = _wilder_smoothing(plus_dm, period)
    minus_dm_sum = _wilder_smoothing(minus_dm, period)
    
    # Avoid division by zero
    plus_di = 100 * plus_dm_sum / np.where(tr_sum > 0, tr_sum, np.nan)
    minus_di = 100 * minus_dm_sum / np.where(tr_sum > 0, tr_sum, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = _wilder_smoothing(dx, period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio_1d_aligned[i]
        adx_val = adx[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_ratio_val) or np.isnan(adx_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + volume spike + ADX > 25
            if close_val > r1_val and vol_ratio_val > 1.5 and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + ADX > 25
            elif close_val < s1_val and vol_ratio_val > 1.5 and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or ADX drops
            if close_val < s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or ADX drops
            if close_val > r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals