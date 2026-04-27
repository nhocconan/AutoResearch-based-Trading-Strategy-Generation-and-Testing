#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and ADX Trend Filter.
Long when price breaks above Donchian upper + ADX > 25 + volume spike.
Short when price breaks below Donchian lower + ADX > 25 + volume spike.
Exit when price returns to Donchian midpoint or ADX < 20.
Designed to generate 20-50 trades/year per symbol with strong edge in trending markets.
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
    
    # Donchian parameters
    donch_len = 20
    adx_len = 14
    
    # Calculate Donchian channels
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donch_len - 1, n):
        upper[i] = np.max(high[i-donch_len+1:i+1])
        lower[i] = np.min(low[i-donch_len+1:i+1])
    
    # Calculate ADX
    # True Range
    tr0 = high - low
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = np.nan
    down_move[0] = np.nan
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    # Initial averages
    if n >= adx_len:
        atr[adx_len-1] = np.nansum(tr[:adx_len])
        plus_dm_sum = np.nansum(plus_dm[:adx_len])
        minus_dm_sum = np.nansum(minus_dm[:adx_len])
        
        for i in range(adx_len, n):
            atr[i] = (atr[i-1] * (adx_len - 1) + tr[i]) / adx_len
            plus_dm_val = (plus_dm_sum * (adx_len - 1) + plus_dm[i]) / adx_len
            minus_dm_val = (minus_dm_sum * (adx_len - 1) + minus_dm[i]) / adx_len
            plus_dm_sum = plus_dm_val * adx_len
            minus_dm_sum = minus_dm_val * adx_len
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_val / atr[i]
                minus_di[i] = 100 * minus_dm_val / atr[i]
            else:
                plus_di[i] = 0
                minus_di[i] = 0
    
    # Calculate ADX
    adx = np.full(n, np.nan)
    if n >= 2 * adx_len - 1:
        dx = np.full(n, np.nan)
        for i in range(adx_len, n):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            else:
                dx[i] = 0
        
        # Initial ADX
        adx[2*adx_len-2] = np.nanmean(dx[adx_len:2*adx_len-1])
        for i in range(2*adx_len-1, n):
            adx[i] = (adx[i-1] * (adx_len - 1) + dx[i]) / adx_len
    
    # Volume filter: volume > 1.8x average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + ADX (2*14-1=27) + volume MA (20)
    start_idx = max(donch_len-1, 2*adx_len-1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        upper_ch = upper[i]
        lower_ch = lower[i]
        adx_val = adx[i]
        
        # Volume filter: volume > 1.8x average
        vol_filter = vol_now > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Bull: price breaks above upper + ADX > 25 + volume spike
            if price_now > upper_ch and adx_val > 25 and vol_filter:
                signals[i] = size
                position = 1
            # Bear: price breaks below lower + ADX > 25 + volume spike
            elif price_now < lower_ch and adx_val > 25 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to midpoint or ADX < 20
            midpoint = (upper_ch + lower_ch) * 0.5
            if price_now < midpoint or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to midpoint or ADX < 20
            midpoint = (upper_ch + lower_ch) * 0.5
            if price_now > midpoint or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_ADX25_Volume"
timeframe = "4h"
leverage = 1.0