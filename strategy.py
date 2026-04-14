#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Donchian channel breakout with volume confirmation
# and 12h ADX trend filter. Long when price breaks above 1-day Donchian high (20-period)
# with volume > 1.5x 20-period average volume AND 12h ADX > 25. Short when price breaks
# below 1-day Donchian low with same conditions. Exit when price returns to the
# 1-day Donchian midpoint or ADX drops below 20. This captures institutional breakouts
# with trend alignment, working in both bull (breakouts continue) and bear (breakdowns
# accelerate) markets. Volume filter ensures breakouts have conviction, reducing false
# signals. Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize
# fee drag while maintaining statistical significance.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Donchian channels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day Donchian channels (20-period)
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    # Middle band: average of upper and lower
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 20-period average volume for confirmation
    volume_series = pd.Series(volume_1d)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Load 12h data ONCE for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14) on 12h data
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values using Wilder's smoothing
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need ADX and Donchian periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        # Note: volume is 6h volume, vol_ma_aligned is 12h average volume scaled to 6h
        # We compare 6h volume to 1.5x the 12h average volume (conceptual, but aligned)
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Weak trend filter: ADX < 20 indicates trend weakening
        weak_trend = adx_aligned[i] < 20
        
        if position == 0:
            # Look for breakout entries with volume and trend confirmation
            # Long: price breaks above Donchian high AND volume confirmation AND strong trend
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm and
                strong_trend):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND volume confirmation AND strong trend
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm and
                  strong_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or trend weakens
            if (close[i] <= donchian_mid_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or trend weakens
            if (close[i] >= donchian_mid_aligned[i] or 
                weak_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Donchian_Breakout_Volume_12hADX_Filter_v1"
timeframe = "6h"
leverage = 1.0