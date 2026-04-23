#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high in 1d uptrend (ADX > 25) with volume > 1.5x 20-period MA.
Short when price breaks below 20-period Donchian low in 1d downtrend (ADX > 25) with volume > 1.5x 20-period MA.
Exit when price crosses the 20-period Donchian midline or opposite band.
Uses 1d HTF for trend regime to avoid whipsaws in ranging markets. Designed for ~30-60 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    period = 14
    tr_sum = wilders_smoothing(tr, period)
    dm_plus_sum = wilders_smoothing(dm_plus, period)
    dm_minus_sum = wilders_smoothing(dm_minus, period)
    
    # Avoid division by zero
    tr_sum[tr_sum == 0] = 1e-10
    
    di_plus = 100 * dm_plus_sum / tr_sum
    di_minus = 100 * dm_minus_sum / tr_sum
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = wilders_smoothing(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_m = (donchian_h + donchian_l) / 2
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need ADX (14+14+14=42 approx) and Donchian20 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_h[i]) or 
            np.isnan(donchian_l[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume filter: 4h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND trending AND volume confirmation
            if close[i] > donchian_h[i] and trending and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND trending AND volume confirmation
            elif close[i] < donchian_l[i] and trending and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses Donchian midline
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below midline
                if close[i] < donchian_m[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above midline
                if close[i] > donchian_m[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dADX_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0