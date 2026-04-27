#!/usr/bin/env python3
"""
1d Donchian Breakout + 1w MA Trend + Volume Filter
Long: Price breaks above Donchian(20) high with 1w EMA50 uptrend and volume > 1.5x avg
Short: Price breaks below Donchian(20) low with 1w EMA50 downtrend and volume > 1.5x avg
Exit: Opposite Donchian breakout or volume < 0.5x avg
Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe
Works in bull (breakouts) and bear (breakdowns) with trend filter
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1w EMA to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA50, and volume MA20
    start_idx = max(donch_len - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filters
        vol_entry_filter = vol_now > 1.5 * vol_avg
        vol_exit_filter = vol_now < 0.5 * vol_avg
        
        if position == 0:
            # Long: break above Donchian high with 1w EMA50 uptrend and volume filter
            if (price > highest_high[i] and 
                price > ema_1w_aligned[i] and vol_entry_filter):
                signals[i] = size
                position = 1
            # Short: break below Donchian low with 1w EMA50 downtrend and volume filter
            elif (price < lowest_low[i] and 
                  price < ema_1w_aligned[i] and vol_entry_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: break below Donchian low OR low volume
            if (price < lowest_low[i] or vol_exit_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: break above Donchian high OR low volume
            if (price > highest_high[i] or vol_exit_filter):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0