#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
# Long when price breaks above Donchian upper, 12h ADX > 25, and volume > 2.0x 20-bar avg.
# Short when price breaks below Donchian lower, 12h ADX > 25, and volume > 2.0x 20-bar avg.
# Exit when price crosses the Donchian midpoint (mean reversion within the channel).
# Donchian channels provide clear breakout levels, ADX filters for trending markets only,
# and volume confirmation reduces false signals. Works in both bull (breakouts) and bear (breakdowns).

name = "6h_Donchian20_12hADX_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 12h ADX for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([0.0]), tr])  # align length
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([np.array([0.0]), dm_plus])
    dm_minus = np.concatenate([np.array([0.0]), dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    atr = wilder_smooth(tr, period)
    dm_plus_smooth = wilder_smooth(dm_plus, period)
    dm_minus_smooth = wilder_smooth(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr > 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = np.zeros_like(dx)
    adx[period*2-2] = np.mean(dx[period-1:period*2-1])  # first ADX value
    for i in range(period*2-1, len(dx)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Donchian(20) channels from primary timeframe
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    midpoint = (upper + lower) / 2
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, donchian_window)  # warmup for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_upper = upper[i]
        curr_lower = lower[i]
        curr_midpoint = midpoint[i]
        curr_adx = adx_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper, ADX > 25 (trending), volume spike
            if (curr_close > curr_upper and 
                curr_adx > 25 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower, ADX > 25 (trending), volume spike
            elif (curr_close < curr_lower and 
                  curr_adx > 25 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price crosses midpoint (mean reversion within channel)
            if curr_close <= curr_midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price crosses midpoint (mean reversion within channel)
            if curr_close >= curr_midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals