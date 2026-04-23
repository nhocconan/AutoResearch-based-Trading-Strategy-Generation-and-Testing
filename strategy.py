#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power + 1d ADX Regime + Volume Spike
Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period MA
Short when Bull Power < 0 AND Bear Power > 0 AND 1d ADX > 25 AND volume > 1.5x 20-period MA
Exit when Elder Ray signals reverse (Bull Power crosses zero) OR ADX < 20 (range) OR volume drops
Uses 1d HTF for ADX regime filter to avoid whipsaws in low ADX environments, Elder Ray for momentum strength.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Calculate 1d ADX for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    tr = np.zeros(len(high_1d))
    
    for i in range(1, len(high_1d)):
        plus_dm[i] = max(0, high_1d[i] - high_1d[i-1])
        minus_dm[i] = max(0, low_1d[i-1] - low_1d[i])
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])  # First value is simple average
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(high_1d) >= period:
        atr_1d = WilderSmooth(tr, period)
        plus_di_1d = 100 * WilderSmooth(plus_dm, period) / atr_1d
        minus_di_1d = 100 * WilderSmooth(minus_dm, period) / atr_1d
        dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
        adx_1d = WilderSmooth(dx_1d, period)
    else:
        adx_1d = np.full(len(high_1d), np.nan)
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 30, 20)  # Elder Ray EMA13, ADX calculation, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        adx_val = adx_1d_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Volume filter: 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        # ADX regime: >25 = trending, <20 = range (with hysteresis)
        if i == start_idx:
            adx_trending = adx_val > 25
            adx_range = adx_val < 20
        else:
            prev_adx_trending = adx_1d_aligned[i-1] > 25
            prev_adx_range = adx_1d_aligned[i-1] < 20
            adx_trending = adx_val > 25 or (prev_adx_trending and adx_val >= 20)
            adx_range = adx_val < 20 or (prev_adx_range and adx_val <= 25)
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND ADX trending AND volume filter
            if bp > 0 and br < 0 and adx_trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 AND Bear Power > 0 AND ADX trending AND volume filter
            elif bp < 0 and br > 0 and adx_trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Bull Power crosses below zero OR ADX goes range OR volume drops
                if bp <= 0 or adx_range or not vol_filter:
                    exit_signal = True
            elif position == -1:
                # Short exit: Bear Power crosses above zero OR ADX goes range OR volume drops
                if br >= 0 or adx_range or not vol_filter:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_Power_1dADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0