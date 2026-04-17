#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian Breakout with 1d Volume Spike and 1d ADX Filter.
Long when price breaks above 4h Donchian High (20) + 1d volume > 2x 20-period avg + ADX > 20.
Short when price breaks below 4h Donchian Low (20) + volume spike + ADX > 20.
Uses tight entry to limit trades (target 20-30/year) and avoid fee drift.
Works in bull (momentum) and bear (ADX filters weak breakouts).
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
    
    # === 4h Donchian Channels (20) ===
    donchian_len = 20
    donchian_high = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    
    # === 1d Data for Volume and ADX ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1d Volume Spike (>2x 20-period avg) ===
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # === 1d ADX (14) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    di_plus = 100 * dm_plus_smooth / atr_14
    di_minus = 100 * dm_minus_smooth / atr_14
    di_sum = di_plus + di_minus
    dx = np.where(di_sum == 0, 0, 100 * np.abs(di_plus - di_minus) / di_sum)
    adx_14 = wilders_smoothing(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(volume_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume spike: current 1d volume > 2x 20-period average
        vol_spike = volume_1d_aligned[i] > vol_ma_20_aligned[i] * 2.0
        
        # Trend filter: ADX > 20 indicates trending market
        trending = adx_1d_aligned[i] > 20.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above Donchian High + volume spike + trending
            if price > donchian_high[i] and vol_spike and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian Low + volume spike + ADX > 20
            elif price < donchian_low[i] and vol_spike and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below Donchian Low
            if price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above Donchian High
            if price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolume2x_ADX20_Breakout"
timeframe = "4h"
leverage = 1.0