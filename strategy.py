#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian(20) breakout + volume confirmation + ADX regime filter.
Long when price breaks above 1d Donchian upper channel with volume > 1.3x 20-period average and ADX(14) > 25.
Short when price breaks below 1d Donchian lower channel with volume > 1.3x 20-period average and ADX(14) > 25.
Exit when price returns to 1d Donchian middle (mean) or ADX < 20 (regime shift to range).
Donchian channels capture volatility-based structure; volume confirms institutional interest; ADX filters choppy markets.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get 1d data for Donchian channels and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper channel = highest high over 20 periods
    # Lower channel = lowest low over 20 periods
    # Middle channel = (upper + lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Calculate 1d ADX(14) for regime filter
    # ADX requires +DI, -DI, and TR
    # TR = max[(high - low), abs(high - prev_close), abs(low - prev_close)]
    # +DM = high - prev_high if high - prev_high > prev_low - low and high - prev_high > 0, else 0
    # -DM = prev_low - low if prev_low - low > high - prev_high and prev_low - low > 0, else 0
    # +DI = 100 * EMA(+DI) / ATR, -DI = 100 * EMA(-DI) / ATR
    # ADX = EMA(|+DI - -DI| / (+DI + -DI))
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # avoid NaN on first element
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close)
    tr3 = np.abs(low_1d - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    dm_plus = np.where((high_1d - prev_high) > (prev_low - low_1d), np.maximum(high_1d - prev_high, 0), 0)
    dm_minus = np.where((prev_low - low_1d) > (high_1d - prev_high), np.maximum(prev_low - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Avoid division by zero
    dm_plus_safe = np.where(dm_plus_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_safe = np.where(dm_minus_smooth == 0, 1e-10, dm_minus_smooth)
    
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper channel with volume and strong trend (ADX > 25)
            if (close[i] > donchian_upper_aligned[i] and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower channel with volume and strong trend (ADX > 25)
            elif (close[i] < donchian_lower_aligned[i] and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle or trend weakens (ADX < 20)
            if (close[i] < donchian_middle_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle or trend weakens (ADX < 20)
            if (close[i] > donchian_middle_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_ADX_Regime"
timeframe = "12h"
leverage = 1.0