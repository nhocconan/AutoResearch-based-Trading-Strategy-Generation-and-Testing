#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) Breakout + 1d ADX Trend Filter + Volume Confirmation
# - Long when price breaks above 20-bar high on 12h, 1d ADX > 25, and volume > 1.5x 20-period average
# - Short when price breaks below 20-bar low on 12h, 1d ADX > 25, and volume > 1.5x 20-period average
# - Exit when price reverses to opposite Donchian level or ADX < 20 (trend weakening)
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First value has no previous close
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(arr[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # Directional Indicators
    plus_di = 100 * dm_plus_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * dm_minus_smooth / (tr_smooth + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = wilder_smooth(dx, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 12h
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        volume = volume_12h[i]
        adx_val = adx_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian band, ADX > 25, volume confirmation
            if price > upper_band and adx_val > 25 and volume > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian band, ADX > 25, volume confirmation
            elif price < lower_band and adx_val > 25 and volume > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower Donchian band OR ADX < 20 (trend weakening)
            if price < lower_band or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper Donchian band OR ADX < 20 (trend weakening)
            if price > upper_band or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0