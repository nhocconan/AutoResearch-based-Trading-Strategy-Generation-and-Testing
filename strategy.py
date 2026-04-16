#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 12h ADX(14) trend filter and volume confirmation.
# Long when price breaks above 12h Donchian(20) upper band AND 12h ADX > 25 AND 4h volume > 1.5x 20-period average.
# Short when price breaks below 12h Donchian(20) lower band AND 12h ADX > 25 AND 4h volume > 1.5x 20-period average.
# Exit when price crosses the 12h Donchian middle band (20-period average).
# Uses discrete position size 0.25. Donchian channels provide clear trend structure, ADX filters for trending markets only.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: Donchian Channels (20) ===
    donchian_upper_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_12h = (donchian_upper_20_12h + donchian_lower_20_12h) / 2.0
    
    # === 12h Indicators: ADX (14) ===
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_14_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_12h = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_12h = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14_12h = 100 * dm_plus_14_12h / atr_14_12h
    di_minus_14_12h = 100 * dm_minus_14_12h / atr_14_12h
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_14_12h - di_minus_14_12h) / (di_plus_14_12h + di_minus_14_12h)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_14_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20_12h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20_12h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_20_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_14_12h)
    
    # === 4h Indicators: Volume average ===
    vol_ma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    # Track position state
    position = 0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(adx_aligned[i])):
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
        vol_ma = vol_ma_20_4h[i]
        
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
            # LONG: Price breaks above Donchian upper AND ADX > 25 AND 4h volume > 1.5x 20-period avg
            if (price > donchian_upper) and (adx > 25) and (vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND ADX > 25 AND 4h volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (adx > 25) and (vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "4h_12hDonchian20_12hADX_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0