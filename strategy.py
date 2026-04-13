#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w EMA(10) trend filter.
# Long: price breaks above Donchian(20) high + volume > 1.5x 1d average volume + price above 1w EMA(10)
# Short: price breaks below Donchian(20) low + volume > 1.5x 1d average volume + price below 1w EMA(10)
# Uses weekly EMA for trend filter to avoid counter-trend trades, volume to confirm breakout strength.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by using weekly EMA as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1-week data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1w EMA(10)
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 10:
        ema_1w[9] = np.mean(close_1w[:10])  # simple average for first value
        for i in range(10, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (10 + 1)) + (ema_1w[i-1] * (1 - 2 / (10 + 1)))
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Donchian(20) on 4h timeframe
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_vol_1d_aligned[i]
        ema = ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average 1d volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: break above Donchian high + above weekly EMA + volume confirmation
            if (price > donch_high[i] and 
                price > ema and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: break below Donchian low + below weekly EMA + volume confirmation
            elif (price < donch_low[i] and 
                  price < ema and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or below weekly EMA
            if (price < donch_low[i] or
                price < ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or above weekly EMA
            if (price > donch_high[i] or
                price > ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_1w_Donchian_Volume_EMA"
timeframe = "4h"
leverage = 1.0