#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d ADX trend filter.
# Donchian channels identify breakouts, volume confirms institutional interest,
# and ADX ensures we only trade in trending markets to avoid whipsaws.
# Designed to work in both bull and bear markets by filtering for strong trends.
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate ADX (14-period) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(len(close_1d))
    minus_dm = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed values
    atr = np.zeros(len(close_1d))
    plus_di = np.zeros(len(close_1d))
    minus_di = np.zeros(len(close_1d))
    
    # Initial values
    atr[13] = np.mean(tr[1:14])
    plus_dm_sum = np.sum(plus_dm[1:14])
    minus_dm_sum = np.sum(minus_dm[1:14])
    
    for i in range(14, len(close_1d)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_dm_sum = plus_dm_sum - plus_dm_sum/14 + plus_dm[i]
        minus_dm_sum = minus_dm_sum - minus_dm_sum/14 + minus_dm[i]
        plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
        minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
    
    # DX and ADX
    dx = np.zeros(len(close_1d))
    adx = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
        else:
            dx[i] = 0
    
    # Initial ADX value
    adx[27] = np.mean(dx[14:28])
    for i in range(28, len(close_1d)):
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        adx_val = adx_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        # ADX filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_val > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + trend
            if price > highest_high[i] and volume_confirm and trend_filter:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + trend
            elif price < lowest_low[i] and volume_confirm and trend_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian middle or breaks below low
            mid_point = (highest_high[i] + lowest_low[i]) / 2
            if price < mid_point or price < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian middle or breaks above high
            mid_point = (highest_high[i] + lowest_low[i]) / 2
            if price > mid_point or price > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0