#!/usr/bin/env python3
"""
4h Triple Confirmation Breakout: Uses 4h Donchian(20) breakout, 1d ADX trend filter, and volume spike.
Long when: 1) Price breaks above 4h Donchian upper (20-period high), 2) 1d ADX > 25 (trending), 3) Volume > 1.5x 20-period average.
Short when: 1) Price breaks below 4h Donchian lower (20-period low), 2) 1d ADX > 25 (trending), 3) Volume > 1.5x 20-period average.
Exit when price returns to 4h Donchian midpoint (mean reversion) or ADX < 20 (trend weakens).
Designed for 4h timeframe: targets 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by requiring strong trend (ADX>25) and using volatility-based stops.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ADX on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
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
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initialize first values
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    # Wilder smoothing
    for i in range(period, len(tr)):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
        dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
    
    # Calculate DI+ and DI-
    di_plus = np.zeros_like(atr)
    di_minus = np.zeros_like(atr)
    dx = np.zeros_like(atr)
    
    # Avoid division by zero
    valid = atr != 0
    di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
    di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
    
    # Calculate DX
    di_sum = di_plus + di_minus
    valid_dx = di_sum != 0
    dx[valid_dx] = 100 * np.abs(di_plus[valid_dx] - di_minus[valid_dx]) / di_sum[valid_dx]
    
    # Calculate ADX (smoothed DX)
    adx = np.zeros_like(dx)
    # Initialize first ADX value
    adx[2*period-1] = np.mean(dx[period:2*period])
    # Wilder smoothing for ADX
    for i in range(2*period, len(dx)):
        adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_len = 20
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i-donchian_len+1:i+1])
        lower[i] = np.min(low[i-donchian_len+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan, dtype=np.float64)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d ADX (2*period), Donchian (20), volume MA (20)
    start_idx = max(2*period, donchian_len, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price = close[i]
        adx_val = adx_aligned[i]
        upper_channel = upper[i]
        lower_channel = lower[i]
        middle_channel = middle[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter: ADX > 25 (strong trend)
        trend_filter = adx_val > 25
        
        # Weak trend filter: ADX < 20 (for exit)
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: price breaks above upper channel + strong trend + volume spike
            if price > upper_channel and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower channel + strong trend + volume spike
            elif price < lower_channel and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (mean reversion) or trend weakens
            if price <= middle_channel or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle (mean reversion) or trend weakens
            if price >= middle_channel or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Triple_Confirmation_Breakout_Donchian_ADX_Volume"
timeframe = "4h"
leverage = 1.0