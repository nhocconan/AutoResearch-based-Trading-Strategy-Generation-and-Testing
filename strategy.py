#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d ADX trend filter + volume confirmation
# Enters long when price breaks above Donchian upper band with 1d ADX > 25 and volume > 1.5x average.
# Enters short when price breaks below Donchian lower band with 1d ADX > 25 and volume > 1.5x average.
# Exits when price returns to Donchian middle band or ADX drops below 20 (trend weakening).
# Uses ADX to filter for trending markets only, avoiding whipsaws in ranging conditions.
# Volume confirms institutional participation in breakouts.
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and fee drag.

name = "4h_Donchian20_1dADX_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d ADX calculation (trend strength) ===
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
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def smooth_values(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + arr[i]
        return smoothed
    
    period = 14
    atr = smooth_values(tr, period)
    plus_di = 100 * smooth_values(plus_dm, period) / np.where(atr > 0, atr, np.nan)
    minus_di = 100 * smooth_values(minus_dm, period) / np.where(atr > 0, atr, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), np.nan)
    adx = smooth_values(dx, period)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 4h Donchian channels (20-period) ===
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Upper and lower bands
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values  # 20 * 4h = ~3.3 days
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        middle_val = donchian_middle[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(upper_val) or np.isnan(lower_val) or 
            np.isnan(middle_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above upper band, strong trend, volume confirmation
            if close_val > upper_val and adx_val > 25 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band, strong trend, volume confirmation
            elif close_val < lower_val and adx_val > 25 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to middle band or trend weakening
            if close_val <= middle_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to middle band or trend weakening
            if close_val >= middle_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals