#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R1/S1 breakout with volume confirmation and ADX trend filter.
# Long when price breaks above Camarilla R1, volume > 1.5x average, and ADX > 25.
# Short when price breaks below Camarilla S1, volume > 1.5x average, and ADX > 25.
# Exit when price crosses back below/above Camarilla pivot point.
# Uses 12h timeframe with daily pivot levels and ADX for trend strength.
# Target: 15-35 trades/year per symbol to stay within frequency limits.
name = "12h_Camarilla_R1_S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels using previous day's OHLC
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    R1 = pivot + (range_prev * 1.1 / 12)
    S1 = pivot - (range_prev * 1.1 / 12)
    
    # Get daily data for ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    atr_1d = wilder_smooth(tr, period)
    # Avoid division by zero
    atr_1d = np.where(atr_1d == 0, np.finfo(float).eps, atr_1d)
    plus_di_1d = 100 * wilder_smooth(plus_dm, period) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm, period) / atr_1d
    dx_denom = plus_di_1d + minus_di_1d
    dx_denom = np.where(dx_denom == 0, np.finfo(float).eps, dx_denom)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / dx_denom
    adx_1d = wilder_smooth(dx_1d, period)
    
    # Align Camarilla levels and ADX to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure volume MA and ADX are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        R1 = R1_aligned[i]
        S1 = S1_aligned[i]
        pivot_pt = pivot_aligned[i]
        adx = adx_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above R1, ADX > 25, volume confirmation
            if price > R1 and adx > 25 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1, ADX > 25, volume confirmation
            elif price < S1 and adx > 25 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below pivot point
            if price < pivot_pt:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above pivot point
            if price > pivot_pt:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals