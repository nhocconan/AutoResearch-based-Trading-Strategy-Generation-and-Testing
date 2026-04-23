#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d ADX regime filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, ADX > 25 (trending), and volume > 1.5x average.
Short when Bear Power < 0, Bull Power > 0, ADX > 25 (trending), and volume > 1.5x average.
Exit when ADX < 20 (range) or power signals reverse.
Elder Ray shows bull/bear strength via EMA13. ADX filters for trending markets only.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull (strong Bull Power) and bear (strong Bear Power) markets via ADX trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA13 and ADX - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for EMA13 and ADX
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate ADX on 1d
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # first value is simple average
        result[period-1] = np.nanmean(data[1:period])  # skip first NaN
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] - (result[i-1]/period) + (data[i]/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d indicators to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d values aligned to 6h)
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_val = ema13_1d_aligned[i]
        adx_val = adx_aligned[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, ADX > 25 (trending), volume spike
            if (bull_power_val > 0 and bear_power_val < 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power > 0, ADX > 25 (trending), volume spike
            elif (bear_power_val < 0 and bull_power_val > 0 and adx_val > 25 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: ADX < 20 (range) or Bear Power >= 0 (bull strength fading)
                if (adx_val < 20 or bear_power_val >= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: ADX < 20 (range) or Bull Power <= 0 (bear strength fading)
                if (adx_val < 20 or bull_power_val <= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dADX_Volume"
timeframe = "6h"
leverage = 1.0