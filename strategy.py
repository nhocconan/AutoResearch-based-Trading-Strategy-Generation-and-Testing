#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) breakout + volume confirmation + 1d ADX trend filter
# Hypothesis: Donchian breakouts with volume confirmation in the direction of daily trend capture
# strong momentum moves while avoiding choppy markets. Works in both bull and bear by
# following the trend direction via ADX filter.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.

name = "12h_donchian20_volume_adx_v14"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr14 = wilder_smoothing(tr, 14)
    dm_plus_14 = wilder_smoothing(dm_plus, 14)
    dm_minus_14 = wilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 > 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        adx[13] = np.nanmean(dx[1:14])
        for i in range(14, len(dx)):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # Align 1d ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 12h
    donchian_period = 20
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(low, np.nan, dtype=float)
    
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Volume confirmation: volume > 1.5x average volume
    vol_ma = np.full_like(volume, np.nan, dtype=float)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_period + 1:i + 1])
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        if adx_aligned[i] <= 25:
            # In chop, stay flat or reduce position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower Donchian or trend weakens
            if close[i] <= lower[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price touches upper Donchian or trend weakens
            if close[i] >= upper[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: volume > 1.5x average
            if vol_ratio[i] > 1.5:
                # Long breakout: price breaks above upper Donchian
                if close[i] > upper[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below lower Donchian
                elif close[i] < lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals