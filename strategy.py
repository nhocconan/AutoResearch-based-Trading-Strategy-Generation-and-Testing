#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Donchian channel breakouts with volume confirmation and ADX trend filter.
# Long when price breaks above 1-day Donchian high (20) with volume > 1.5x 20-period average and ADX > 25.
# Short when price breaks below 1-day Donchian low (20) with volume > 1.5x 20-period average and ADX > 25.
# Exit when price returns to the 1-day Donchian midpoint or ADX drops below 20.
# Donchian channels provide clear support/resistance levels; volume confirms breakout strength.
# ADX filter ensures trading only in trending conditions, avoiding whipsaws in ranges.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels, volume average, and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:  # Need enough for Donchian(20) and ADX(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period)
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    # Middle band: average of upper and lower
    lookback = 20
    upper_band = np.full_like(high_1d, np.nan)
    lower_band = np.full_like(low_1d, np.nan)
    mid_band = np.full_like(close_1d, np.nan)
    
    for i in range(lookback - 1, len(high_1d)):
        upper_band[i] = np.max(high_1d[i - lookback + 1:i + 1])
        lower_band[i] = np.min(low_1d[i - lookback + 1:i + 1])
        mid_band[i] = (upper_band[i] + lower_band[i]) / 2.0
    
    # Calculate average volume (20-period)
    vol_ma = np.full_like(volume_1d, np.nan)
    for i in range(lookback - 1, len(volume_1d)):
        vol_ma[i] = np.mean(volume_1d[i - lookback + 1:i + 1])
    
    # Calculate ADX (14)
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Smoothed values using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value: simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_14 = wilders_smoothing(tr, 14)
    dm_plus_14 = wilders_smoothing(dm_plus, 14)
    dm_minus_14 = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Align indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    mid_band_aligned = align_htf_to_ltf(prices, df_1d, mid_band)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(40, 34)  # Need Donchian(20), volume MA(20), and ADX(14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for breakout entries in strong trend with volume confirmation
            # Long: price breaks above upper band AND volume confirmation AND strong trend
            if (close[i] > upper_band_aligned[i] and 
                vol_confirm and 
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band AND volume confirmation AND strong trend
            elif (close[i] < lower_band_aligned[i] and 
                  vol_confirm and 
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or trend weakens
            if (close[i] <= mid_band_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to midpoint or trend weakens
            if (close[i] >= mid_band_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Breakout_Volume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0