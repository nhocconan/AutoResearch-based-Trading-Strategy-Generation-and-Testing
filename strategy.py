#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20-period) with 1d ADX filter (>25) and volume confirmation (1.5x).
# Long when price breaks above 4h Donchian upper, 1d ADX > 25, and 1h volume > 1.5x 20-bar median volume.
# Short when price breaks below 4h Donchian lower, 1d ADX > 25, and 1h volume > 1.5x 20-bar median volume.
# Exit when price returns to 4h Donchian middle (mean of upper/lower).
# Uses discrete position size 0.20. Target: 60-150 total trades over 4 years (15-37/year).
# Works in both bull and bear markets by using 1d ADX as a trend filter to avoid ranging markets and Donchian for breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian upper (20-period high)
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Calculate 4h Donchian lower (20-period low)
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Calculate 4h Donchian middle (mean of upper/lower)
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: ADX (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
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
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / atr_1d
    di_minus = 100 * dm_minus_14 / atr_1d
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 1h data for volume filter
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (1h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 20)  # Donchian(20), ADX(14), volume median(20)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        adx = adx_1d_aligned[i]
        vol_median = vol_median_20[i]
        price = close[i]
        vol = volume[i]
        
        # Volume spike filter: current 1h volume > 1.5x median volume
        volume_spike = vol > (vol_median * 1.5)
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price returns to Donchian middle
            if price <= middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price returns to Donchian middle
            if price >= middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: 1d ADX > 25 (strong trend)
            strong_trend = adx > 25
            
            # LONG CONDITIONS
            # Price breaks above 4h Donchian upper, strong trend, and volume spike
            if price > upper and strong_trend and volume_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT CONDITIONS
            # Price breaks below 4h Donchian lower, strong trend, and volume spike
            elif price < lower and strong_trend and volume_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_Donchian20_4hUpperLower_1dADX25_1hVolumeSpike1.5x_v1"
timeframe = "1h"
leverage = 1.0