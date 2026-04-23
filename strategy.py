#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX < 20 (range) AND volume > 1.5x 20-period average.
Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX < 20 (range) AND volume > 1.5x 20-period average.
Exit when power signals reverse or ADX > 25 (trending regime).
Uses 1d HTF for ADX regime to avoid whipsaws in strong trends. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA13 for Elder Ray Power (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ADX for regime filter (HTF)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First element NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR and DM (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_period = 14
    atr1d = wilders_smoothing(tr, atr_period)
    plus_dm1d = wilders_smoothing(plus_dm, atr_period)
    minus_dm1d = wilders_smoothing(minus_dm, atr_period)
    
    # DI and DX
    plus_di1d = np.where(atr1d != 0, (plus_dm1d / atr1d) * 100, 0)
    minus_di1d = np.where(atr1d != 0, (minus_dm1d / atr1d) * 100, 0)
    dx1d = np.where((plus_di1d + minus_di1d) != 0, 
                    np.abs((plus_di1d - minus_di1d) / (plus_di1d + minus_di1d)) * 100, 0)
    
    # ADX (smoothed DX)
    adx1d = wilders_smoothing(dx1d, atr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx1d)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 20)  # Elder Ray (13), volume MA (20), ADX (14+14=28)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX < 20 (range) AND volume spike
            if bull_power > 0 and bear_power < 0 and adx_val < 20 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND ADX < 20 (range) AND volume spike
            elif bear_power > 0 and bull_power < 0 and adx_val < 20 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR ADX > 25 (trending)
                if bull_power <= 0 or bear_power >= 0 or adx_val > 25:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 OR Bull Power >= 0 OR ADX > 25 (trending)
                if bear_power <= 0 or bull_power >= 0 or adx_val > 25:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Power_1dADX_Range_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0