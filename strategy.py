#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power (Bull/Bear) with 1d ADX regime filter and volume confirmation.
- Bull Power = High - EMA13(close); Bear Power = EMA13(close) - Low
- Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
- Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period average
- Exit when power signals weaken (Bull Power <= 0 for long, Bear Power <= 0 for short) OR ADX < 20 (range) OR volume normalizes
- Uses 1d HTF for ADX regime to avoid whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
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
    tr = np.concatenate([[np.nan], tr])  # align with 1d index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values (Wilder's smoothing = EMA with alpha=1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(x):
        result = np.full_like(x, np.nan)
        for i in range(len(x)):
            if np.isnan(x[i]):
                continue
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = x[i]
            else:
                result[i] = alpha * x[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr)
    atr_period = wilders_smoothing(atr)  # ATR smoothed again for ADX calculation
    
    di_plus = 100 * wilders_smoothing(dm_plus) / atr_period
    di_minus = 100 * wilders_smoothing(dm_minus) / atr_period
    
    dx = np.abs(di_plus - di_minus) / (np.abs(di_plus) + np.abs(di_minus)) * 100
    dx = np.where((np.abs(di_plus) + np.abs(di_minus)) == 0, 0, dx)
    
    adx = wilders_smoothing(dx)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate EMA13 for Elder Ray Power (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30 + 14*3, 20)  # EMA13, ADX calculation buffers, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bull_power[i]
        bp_bear = bear_power[i]
        adx_val = adx_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (trending) AND volume spike
            if bp > 0 and bp_bear < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (trending) AND volume spike
            elif bp_bear > 0 and bp < 0 and adx_val > 25 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (range) OR volume normalizes
                if bp <= 0 or bp_bear >= 0 or adx_val < 20 or volume[i] <= vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power <= 0 OR Bull Power >= 0 OR ADX < 20 (range) OR volume normalizes
                if bp_bear <= 0 or bp >= 0 or adx_val < 20 or volume[i] <= vol_ma_val:
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