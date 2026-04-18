#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ADXTrend
Hypothesis: Donchian(20) breakouts with volume spike and ADX trend filter.
Buy when price breaks above upper band with volume spike and ADX > 25.
Sell when price breaks below lower band with volume spike and ADX > 25.
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing
strong momentum moves in trending markets. Works in both bull and bear markets
by only taking trades in the direction of strong trends (ADX > 25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX on 1d timeframe
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
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr14 = wilder_smooth(tr, 14)
    dm_plus_14 = wilder_smooth(dm_plus, 14)
    dm_minus_14 = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(tr14 != 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 != 0, 100 * dm_minus_14 / tr14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 4h timeframe (wait for daily bar to close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        adx_val = adx_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and strong trend (ADX > 25)
            if price > upper_val and vol_spike and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and strong trend (ADX > 25)
            elif price < lower_val and vol_spike and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns below upper band OR trend weakens (ADX < 20)
            if price < upper_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns above lower band OR trend weakens (ADX < 20)
            if price > lower_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0