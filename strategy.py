#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter + volume confirmation.
Long when Bull Power > 0, 1d ADX > 25 (trending), and volume > 1.5x 20-period average.
Short when Bear Power < 0, 1d ADX > 25 (trending), and volume > 1.5x 20-period average.
Exit when Elder Power reverses sign or ADX < 20 (range) or volume normalizes.
Uses 1d HTF for ADX regime to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years (19-50/year).
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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 1d ADX for regime filter (HTF)
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmooth(tr, 14)
    plus_dm_smooth = WilderSmooth(plus_dm, 14)
    minus_dm_smooth = WilderSmooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = WilderSmooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray Power (Bull Power = High - EMA13, Bear Power = Low - EMA13)
    bull_power = high - ema13_aligned
    bear_power = low - ema13_aligned
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # ADX (30), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0, ADX > 25 (trending), volume spike
            if bull_val > 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, ADX > 25 (trending), volume spike
            elif bear_val < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR ADX < 20 (range) OR volume normalizes
                if bull_val <= 0 or adx_val < 20 or volume[i] <= vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power >= 0 OR ADX < 20 (range) OR volume normalizes
                if bear_val >= 0 or adx_val < 20 or volume[i] <= vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Power_1dADX_Regime_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0