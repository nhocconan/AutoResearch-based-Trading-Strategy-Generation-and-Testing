#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d ADX regime filter and volume confirmation.
Long when Bear Power < 0 (bulls in control) AND ADX > 25 (trending) AND volume > 1.5x 20-period MA.
Short when Bull Power > 0 (bears in control) AND ADX > 25 AND volume > 1.5x 20-period MA.
Exit when Elder Power reverses sign OR ADX < 20 (range) OR volume drops.
Uses 1d HTF for ADX regime filter to avoid whipsaws in low-volatility environments, Elder Ray for precise
momentum measurement of bull/bear power relative to EMA13. Target: 50-150 total trades over 4 years.
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
    
    # Calculate 1d EMA13 for Elder Ray (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 1d ADX for regime filter (HTF)
    # TR = max[(H-L), abs(H-PC), abs(L-PC)]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # +DM = max(H-PH, 0) if H-PH > PL-L else 0
    # -DM = max(PL-L, 0) if PL-L > H-PH else 0
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    period = 14
    tr14 = wilders_smoothing(tr, period)
    dm_plus14 = wilders_smoothing(dm_plus, period)
    dm_minus14 = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, (dm_plus14 / tr14) * 100, 0)
    di_minus = np.where(tr14 != 0, (dm_minus14 / tr14) * 100, 0)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    
    # ADX = smoothed DX
    adx_1d = wilders_smoothing(dx, period)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA13 smoothing + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: 6h volume > 1.5x 20-period MA (moderate threshold)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # ADX regime filter: >25 for trending, <20 for range (with hysteresis)
        if i >= start_idx + 1:
            adx_prev = adx_aligned[i-1]
            adx_trending = adx_aligned[i] > 25
            adx_ranging = adx_aligned[i] < 20
            # Hysteresis: once trending, stay trending until ADX < 20
            if position != 0:
                # In a position, only exit if ADX drops below 20
                adx_regime_ok = not adx_ranging
            else:
                # Looking to enter, need ADX > 25
                adx_regime_ok = adx_trending
        else:
            adx_regime_ok = False
        
        if position == 0:
            # Long: Bear Power < 0 (bulls in control) AND ADX trending AND volume filter
            if bear_power_aligned[i] < 0 and adx_regime_ok and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power > 0 (bears in control) AND ADX trending AND volume filter
            elif bull_power_aligned[i] > 0 and adx_regime_ok and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bear Power >= 0 OR ADX becomes ranging OR volume drops
                if (bear_power_aligned[i] >= 0 or 
                    (i >= start_idx + 1 and adx_aligned[i] < 20 and adx_aligned[i-1] >= 20) or
                    not vol_filter):
                    exit_signal = True
            elif position == -1:
                # Short exit: Bull Power <= 0 OR ADX becomes ranging OR volume drops
                if (bull_power_aligned[i] <= 0 or 
                    (i >= start_idx + 1 and adx_aligned[i] < 20 and adx_aligned[i-1] >= 20) or
                    not vol_filter):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0