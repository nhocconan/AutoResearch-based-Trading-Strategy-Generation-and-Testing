#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Donchian breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 1w Donchian(20) upper band AND 1d ADX > 25 AND 12h volume > 1.5x 20-period average.
# Short when price breaks below 1w Donchian(20) lower band AND 1d ADX > 25 AND 12h volume > 1.5x 20-period average.
# Exit when price crosses the 1w Donchian middle band (20-period average).
# Uses discrete position size 0.25. Donchian channels provide clear trend structure, ADX filters for trending markets only.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 12h data once before loop for volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # === 1w Indicators: Donchian Channels (20) ===
    # Donchian upper band = highest high over 20 periods
    donchian_upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian lower band = lowest low over 20 periods
    donchian_lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    # Donchian middle band = average of upper and lower
    donchian_middle_20_1w = (donchian_upper_20_1w + donchian_lower_20_1w) / 2.0
    
    # === 1d Indicators: ADX (14) ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_1d = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_1d = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus_14_1d = 100 * dm_plus_14_1d / atr_14_1d
    di_minus_14_1d = 100 * dm_minus_14_1d / atr_14_1d
    
    # DX and ADX
    dx_14_1d = 100 * np.abs(di_plus_14_1d - di_minus_14_1d) / (di_plus_14_1d + di_minus_14_1d)
    dx_14_1d[di_plus_14_1d + di_minus_14_1d == 0] = 0  # avoid division by zero
    adx_14_1d = pd.Series(dx_14_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 12h Indicators: Volume average ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_20_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        adx = adx_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 12h volume aligned
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        vol_12h_current = vol_12h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            if price < donchian_middle:  # Exit when price crosses below middle band
                exit_signal = True
        
        elif position == -1:  # Short position
            if price > donchian_middle:  # Exit when price crosses above middle band
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND ADX > 25 AND 12h volume > 1.5x 20-period avg
            if (price > donchian_upper) and (adx > 25) and (vol_12h_current > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND ADX > 25 AND 12h volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (adx > 25) and (vol_12h_current > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1wDonchian20_1dADX_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0