#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and 1d ADX trend filter.
# Long when price breaks above 12h Donchian upper band (20) with volume > 1.5x 20-period average and 1d ADX > 25.
# Short when price breaks below 12h Donchian lower band (20) with volume > 1.5x 20-period average and 1d ADX > 25.
# Exit when price returns to 12h Donchian middle (midpoint of upper/lower).
# Uses Donchian channels for structure, volume surge for conviction, ADX for trend strength filter.
# Designed for ~20-30 trades/year per symbol.
name = "12h_Donchian_20_Volume_ADX_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def smooth_wilder(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1d = smooth_wilder(tr, 14)
    dm_plus_smooth = smooth_wilder(dm_plus, 14)
    dm_minus_smooth = smooth_wilder(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d > 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = smooth_wilder(dx, 14)
    
    # Align ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12h Donchian channels (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    donch_mid = np.zeros(n)
    
    for i in range(n):
        if i >= 19:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
            donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
        else:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
            donch_mid[i] = np.nan
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        if i >= 19:
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        donch_mid_val = donch_mid[i]
        adx_val = adx_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume surge and strong trend
            if close_val > donch_high_val and vol_filter and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume surge and strong trend
            elif close_val < donch_low_val and vol_filter and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to Donchian middle
            if close_val <= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to Donchian middle
            if close_val >= donch_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals