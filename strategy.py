#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1w Vortex trend filter and volume confirmation.
# Long when price breaks above 1d Donchian(20) upper band AND 1w VI+ > VI- AND 4h volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian(20) lower band AND 1w VI- > VI+ AND 4h volume > 1.5x 20-period average.
# Exit when price crosses the 1d Donchian middle band (20-period average).
# Uses discrete position size 0.25. Donchian channels provide clear trend structure, Vortex filters for trending markets only.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 75-200 trades over 4 years (19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for Vortex
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 4h data once before loop for volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    
    # === 1d Indicators: Donchian Channels (20) ===
    donchian_upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1d = (donchian_upper_20_1d + donchian_lower_20_1d) / 2.0
    
    # === 1w Indicators: Vortex (14) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Vortex Indicator
    vm_plus = np.abs(high_1w - np.roll(low_1w, 1))
    vm_minus = np.abs(low_1w - np.roll(high_1w, 1))
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Smoothed TR, VM+, VM- (Wilder's smoothing)
    atr_14_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vm_plus_14_1w = pd.Series(vm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    vm_minus_14_1w = pd.Series(vm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # VI+ and VI-
    vi_plus_14_1w = vm_plus_14_1w / atr_14_1w
    vi_minus_14_1w = vm_minus_14_1w / atr_14_1w
    
    # === 4h Indicators: Volume average ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_20_1d)
    vi_plus_aligned = align_htf_to_ltf(prices, df_1w, vi_plus_14_1w)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1w, vi_minus_14_1w)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        vi_plus = vi_plus_aligned[i]
        vi_minus = vi_minus_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Get 4h volume aligned
        vol_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        vol_4h_current = vol_4h_aligned[i]
        
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
            # LONG: Price breaks above Donchian upper AND VI+ > VI- AND 4h volume > 1.5x 20-period avg
            if (price > donchian_upper) and (vi_plus > vi_minus) and (vol_4h_current > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND VI- > VI+ AND 4h volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (vi_minus > vi_plus) and (vol_4h_current > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dDonchian20_1wVortex_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0