#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ADX filter + volume confirmation
# - Donchian breakout on 12h provides directional trend signals
# - 1d ADX > 25 ensures we only trade in trending markets (avoids whipsaw)
# - Volume > 1.5x 20-period average confirms breakout strength
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/14)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + alpha * data[i]
        return result
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    adx_1d = adx  # Already smoothed
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian(20) on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    highest_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_12h / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        highest_high = highest_high_20[i]
        lowest_low = lowest_low_20[i]
        adx_val = adx_1d_aligned[i]
        vol_ratio = volume_ratio[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high + ADX > 25 + volume confirmation
            if price > highest_high and adx_val > 25 and vol_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low + ADX > 25 + volume confirmation
            elif price < lowest_low and adx_val > 25 and vol_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or ADX falls below 20
            if price < lowest_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or ADX falls below 20
            if price > highest_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dADX_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0