#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout (20) with volume confirmation and 12h ADX trend filter.
# Donchian breakouts capture breakout momentum. Volume surge confirms institutional participation.
# 12h ADX > 25 ensures trades occur in trending markets, avoiding whipsaws in chop.
# Works in bull/bear by following strong trends from volatility contractions.
# Target: 100-200 total trades over 4 years (25-50/year). Size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Donchian Channel (20) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian channels
    dc_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    dc_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    dc_upper_aligned = align_htf_to_ltf(prices, df_12h, dc_upper)
    dc_lower_aligned = align_htf_to_ltf(prices, df_12h, dc_lower)
    
    # === 12h ADX (14) for trend strength ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate +DM and -DM
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_14 = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    di_plus = 100 * dm_plus_smooth / atr_14
    di_minus = 100 * dm_minus_smooth / atr_14
    
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx_14 = wilders_smoothing(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_14)
    
    # === 4h volume for surge confirmation ===
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(dc_upper_aligned[i]) or 
            np.isnan(dc_lower_aligned[i]) or
            np.isnan(adx_12h_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume surge: current 4h volume > 1.5x 20-period average
        vol_4h_current = volume[i]  # Current 4h volume from primary timeframe
        vol_surge = vol_4h_current > vol_ma_20_aligned[i] * 1.5
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25.0
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above 12h DC upper + volume surge + trending
            if price > dc_upper_aligned[i] and vol_surge and trending:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below 12h DC lower + volume surge + trending
            elif price < dc_lower_aligned[i] and vol_surge and trending:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below 12h DC lower
            if price < dc_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above 12h DC upper
            if price > dc_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hVolume1.5x_12hADX25_TrendFilter"
timeframe = "4h"
leverage = 1.0