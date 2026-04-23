#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND Bull Power > 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average.
Exit when ADX < 20 (range) OR power signs flip OR volume drops.
Uses 1d HTF for ADX to avoid 6h whipsaws. Target: 50-150 total trades over 4 years (12-37/year).
Elder Ray: Bull Power = high - EMA13, Bear Power = low - EMA13. ADX measures trend strength.
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
    
    # Calculate 13-period EMA for Elder Ray (using 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13  # Bull Power = High - EMA
    bear_power = low - ema_13   # Bear Power = Low - EMA
    
    # Calculate 1d ADX for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing: alpha = 1/period"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
            else:
                result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX = smoothed DX
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 20-period volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20, 30 + 14 + 14)  # EMA13, vol MA, ADX calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending) AND volume spike
            if bp > 0 and br < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 AND ADX > 25 (trending) AND volume spike
            elif br < 0 and bp > 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: ADX < 20 (ranging) OR Bull Power <= 0 OR Bear Power >= 0 OR volume drop
                if adx_val < 20 or bp <= 0 or br >= 0 or volume[i] < 0.8 * vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: ADX < 20 (ranging) OR Bear Power >= 0 OR Bull Power <= 0 OR volume drop
                if adx_val < 20 or br >= 0 or bp <= 0 or volume[i] < 0.8 * vol_ma_val:
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