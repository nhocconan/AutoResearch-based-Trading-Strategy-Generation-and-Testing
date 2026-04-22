#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX25 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band with 1d ADX>25 (trending) and volume spike
# Short when price breaks below 4h Donchian lower band with 1d ADX>25 and volume spike
# Uses Donchian channel as price structure, ADX for trend strength filter, volume for confirmation
# Designed for 4h timeframe to target 20-40 trades/year per symbol.
# Focus on strong trending moves with volume confirmation to avoid whipsaws and reduce trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for ADX trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def _wilder_smoothing(x, period):
        result = np.full_like(x, np.nan, dtype=float)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period]) if np.any(~np.isnan(x[1:period])) else 0
        # Wilder smoothing: today = (yesterday * (period-1) + today) / period
        for i in range(period, len(x)):
            if np.isnan(result[i-1]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + x[i]) / period
        return result
    
    atr = _wilder_smoothing(tr, 14)
    dm_plus_smooth = _wilder_smoothing(dm_plus, 14)
    dm_minus_smooth = _wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = _wilder_smoothing(dx, 14)
    
    adx_25 = adx > 25
    
    # Align ADX to 4h timeframe
    adx_25_aligned = align_htf_to_ltf(prices, df_1d, adx_25.astype(float))
    
    # Donchian channel (20-period) on 4h data
    def _rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmax(arr[i-window+1:i+1])
        return result
    
    def _rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            result[i] = np.nanmin(arr[i-window+1:i+1])
        return result
    
    donchian_high = _rolling_max(high, 20)
    donchian_low = _rolling_min(low, 20)
    
    # Volume spike filter (20-period on 4h data)
    vol_ma20 = np.full_like(volume, np.nan, dtype=float)
    for i in range(19, len(volume)):
        vol_ma20[i] = np.nanmean(volume[i-19:i+1])
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_25_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + ADX>25 + volume spike
            if (close[i] > donchian_high[i] and 
                adx_25_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + ADX>25 + volume spike
            elif (close[i] < donchian_low[i] and 
                  adx_25_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Donchian band or ADX < 20 (trend weakening)
            if position == 1:
                if close[i] < donchian_low[i] or not adx_25_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i] or not adx_25_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX25_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0