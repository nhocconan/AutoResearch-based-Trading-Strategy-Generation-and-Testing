#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 4h ADX(14) trend filter and volume confirmation.
# Long when price breaks above 1d Donchian(20) upper band AND 4h ADX > 25 AND 4h volume > 1.5x 20-period average.
# Short when price breaks below 1d Donchian(20) lower band AND 4h ADX > 25 AND 4h volume > 1.5x 20-period average.
# Exit when price crosses the 1d Donchian middle band (20-period average).
# Uses discrete position size 0.25. 1d/4h filters provide signal direction, 4h provides entry timing.
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for ADX and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: Donchian Channels (20) ===
    donchian_upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1d = (donchian_upper_20_1d + donchian_lower_20_1d) / 2.0
    
    # === 4h Indicators: ADX (14) ===
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_14_4h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_4h = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_4h = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14_4h = 100 * dm_plus_14_4h / atr_14_4h
    di_minus_14_4h = 100 * dm_minus_14_4h / atr_14_4h
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_14_4h - di_minus_14_4h) / (di_plus_14_4h + di_minus_14_4h)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_14_4h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 4h Indicators: Volume MA (20) ===
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_20_1d)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_14_4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        adx = adx_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_aligned[i]
        
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
            # LONG: Price breaks above Donchian upper AND ADX > 25 AND volume > 1.5x 20-period avg
            if (price > donchian_upper) and (adx > 25) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND ADX > 25 AND volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (adx > 25) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_1dDonchian20_4hADX_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0