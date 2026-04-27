#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above Donchian(20) upper band with 1d ADX > 25 and volume > 1.5x average.
# Short when price breaks below Donchian(20) lower band with 1d ADX > 25 and volume > 1.5x average.
# Exit when price crosses back through Donchian midline.
# Uses price channel breakouts with trend filter to avoid whipsaw in sideways markets.
# Target: 20-50 trades per year on 4h timeframe.

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
    if len(df_1d) < 25:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
    adx_period = 14
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    # Directional movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR, +DM, -DM
    def smooth_wilder(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])  # Skip first NaN
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    tr_smooth = smooth_wilder(tr, adx_period)
    plus_dm_smooth = smooth_wilder(plus_dm, adx_period)
    minus_dm_smooth = smooth_wilder(minus_dm, adx_period)
    
    # Calculate +DI and -DI
    plus_di = np.full_like(tr_smooth, np.nan)
    minus_di = np.full_like(tr_smooth, np.nan)
    mask = tr_smooth != 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.full_like(tr_smooth, np.nan)
    di_sum = plus_di + minus_di
    mask = di_sum != 0
    dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    
    adx = np.full_like(tr_smooth, np.nan)
    if len(dx) >= adx_period:
        # First ADX is average of first 'adx_period' DX values
        start_idx = adx_period - 1
        if not np.isnan(dx[start_idx]):
            adx[start_idx] = np.nanmean(dx[1:start_idx+1])  # Skip first NaN
            for i in range(start_idx + 1, len(dx)):
                if not np.isnan(adx[i-1]) and not np.isnan(dx[i]):
                    adx[i] = adx[i-1] - (adx[i-1] / adx_period) + dx[i]
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period)
    donch_period = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        highest_high[i] = np.max(high[i - donch_period + 1:i + 1])
        lowest_low[i] = np.min(low[i - donch_period + 1:i + 1])
    
    # Donchian midline
    donch_mid = (highest_high + lowest_low) / 2
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, ADX, and volume MA20
    start_idx = max(donch_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # ADX filter: require trending market
        adx_filter = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian upper band with ADX > 25 and volume filter
            if (price > highest_high[i] and adx_filter and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian lower band with ADX > 25 and volume filter
            elif (price < lowest_low[i] and adx_filter and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if price < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if price > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0