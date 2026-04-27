#!/usr/bin/env python3
"""
1d_Price_Action_With_Volume_Regime
Price action strategy using 1d timeframe with volume confirmation and regime filter.
Long when price breaks above Donchian high (20) with volume > 1.5x average and choppy regime.
Short when price breaks below Donchian low (20) with volume > 1.5x average and choppy regime.
Exit when price crosses opposite Donchian boundary or regime shifts to trending.
Uses 1w trend filter (ADX) to avoid counter-trend trades in strong trends.
Target: 10-25 trades/year per symbol.
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
    
    # Donchian channels (20-period)
    donch_len = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(donch_len - 1, n):
        highest_high[i] = np.max(high[i - donch_len + 1:i + 1])
        lowest_low[i] = np.min(low[i - donch_len + 1:i + 1])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(19, n):
        avg_volume[i] = np.mean(volume[i - 19:i + 1])
    
    # Choppiness Index (14-period) for regime filter
    chop_len = 14
    atr = np.full(n, np.nan)
    for i in range(1, n):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr[i] = tr if i == 1 else (atr[i-1] * 13 + tr) / 14  # Wilder's smoothing
    
    sum_atr = np.full(n, np.nan)
    for i in range(chop_len - 1, n):
        if i >= chop_len - 1:
            sum_atr[i] = np.sum(atr[i - chop_len + 1:i + 1])
    
    highest_high_chop = np.full(n, np.nan)
    lowest_low_chop = np.full(n, np.nan)
    for i in range(chop_len - 1, n):
        highest_high_chop[i] = np.max(high[i - chop_len + 1:i + 1])
        lowest_low_chop[i] = np.min(low[i - chop_len + 1:i + 1])
    
    chop = np.full(n, np.nan)
    for i in range(chop_len - 1, n):
        if sum_atr[i] > 0 and (highest_high_chop[i] - lowest_low_chop[i]) > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (highest_high_chop[i] - lowest_low_chop[i])) / np.log10(chop_len)
        else:
            chop[i] = 50  # neutral
    
    # Get 1w data for higher timeframe trend filter (ADX)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14-period) on 1w
    adx_len = 14
    tr_1w = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    dm_plus = np.full(len(close_1w), np.nan)
    dm_minus = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        up = high_1w[i] - high_1w[i-1]
        down = low_1w[i-1] - low_1w[i]
        dm_plus[i] = up if up > down and up > 0 else 0
        dm_minus[i] = down if down > up and down > 0 else 0
    
    # Smoothed values
    tr_sum = np.full(len(close_1w), np.nan)
    dm_plus_sum = np.full(len(close_1w), np.nan)
    dm_minus_sum = np.full(len(close_1w), np.nan)
    
    for i in range(adx_len - 1, len(close_1w)):
        if i == adx_len - 1:
            tr_sum[i] = np.sum(tr_1w[i - adx_len + 1:i + 1])
            dm_plus_sum[i] = np.sum(dm_plus[i - adx_len + 1:i + 1])
            dm_minus_sum[i] = np.sum(dm_minus[i - adx_len + 1:i + 1])
        else:
            tr_sum[i] = tr_sum[i-1] - tr_sum[i-1]/adx_len + tr_1w[i]
            dm_plus_sum[i] = dm_plus_sum[i-1] - dm_plus_sum[i-1]/adx_len + dm_plus[i]
            dm_minus_sum[i] = dm_minus_sum[i-1] - dm_minus_sum[i-1]/adx_len + dm_minus[i]
    
    di_plus = np.full(len(close_1w), np.nan)
    di_minus = np.full(len(close_1w), np.nan)
    dx = np.full(len(close_1w), np.nan)
    for i in range(adx_len - 1, len(close_1w)):
        if tr_sum[i] > 0:
            di_plus[i] = 100 * dm_plus_sum[i] / tr_sum[i]
            di_minus[i] = 100 * dm_minus_sum[i] / tr_sum[i]
            dx[i] = 100 * abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        else:
            di_plus[i] = 0
            di_minus[i] = 0
            dx[i] = 0
    
    adx = np.full(len(close_1w), np.nan)
    for i in range(2*adx_len - 2, len(close_1w)):
        if i == 2*adx_len - 2:
            adx[i] = np.sum(dx[i - adx_len + 1:i + 1]) / adx_len
        else:
            adx[i] = (adx[i-1] * (adx_len - 1) + dx[i]) / adx_len
    
    # Align 1w ADX to 1d timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, volume avg, chop, and ADX
    start_idx = max(donch_len - 1, 19, chop_len - 1, 2*adx_len - 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(chop[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        chop_value = chop[i]
        adx_value = adx_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike in choppy regime
            if (price > highest_high[i] and vol > 1.5 * avg_vol and 
                chop_value > 61.8 and adx_value < 25):
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with volume spike in choppy regime
            elif (price < lowest_low[i] and vol > 1.5 * avg_vol and 
                  chop_value > 61.8 and adx_value < 25):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian low or regime shifts to trending
            if (price < lowest_low[i] or chop_value < 38.2 or adx_value > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian high or regime shifts to trending
            if (price > highest_high[i] or chop_value < 38.2 or adx_value > 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Price_Action_With_Volume_Regime"
timeframe = "1d"
leverage = 1.0