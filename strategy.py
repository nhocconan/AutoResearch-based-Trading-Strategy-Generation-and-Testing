#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian(20) breakout with 1w ADX(14) trend filter and volume confirmation.
# Long when price breaks above 1w Donchian(20) upper band AND 1w ADX > 25 AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 1w Donchian(20) lower band AND 1w ADX > 25 AND 1d volume > 1.5x 20-period average.
# Exit when price crosses the 1w Donchian middle band (20-period average).
# Uses discrete position size 0.25. 1w filters provide signal direction, 1d provides entry timing.
# Target: 30-100 total trades over 4 years (7-25/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Donchian channels and ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Donchian Channels (20) ===
    donchian_upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_middle_20_1w = (donchian_upper_20_1w + donchian_lower_20_1w) / 2.0
    
    # === 1w Indicators: ADX (14) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    atr_14_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14_1w = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14_1w = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus_14_1w = 100 * dm_plus_14_1w / atr_14_1w
    di_minus_14_1w = 100 * dm_minus_14_1w / atr_14_1w
    
    # DX and ADX
    dx = 100 * np.abs(di_plus_14_1w - di_minus_14_1w) / (di_plus_14_1w + di_minus_14_1w)
    dx = np.where(np.isnan(dx), 0, dx)
    adx_14_1w = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20_1w)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20_1w)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1w, donchian_middle_20_1w)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_14_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
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
        
        # Get 1d volume average
        vol_ma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        
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
            if (price > donchian_upper) and (adx > 25) and (vol > 1.5 * vol_ma_20_1d[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND ADX > 25 AND volume > 1.5x 20-period avg
            elif (price < donchian_lower) and (adx > 25) and (vol > 1.5 * vol_ma_20_1d[i]):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wDonchian20_1wADX_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0