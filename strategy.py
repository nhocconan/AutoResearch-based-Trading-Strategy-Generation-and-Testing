#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1w and 1d confluence.
# Long: Price above 1w EMA50 (trend filter) + breaks above 1d Donchian upper channel + volume > 1.5x 20-period average.
# Short: Price below 1w EMA50 + breaks below 1d Donchian lower channel + volume > 1.5x average.
# Uses 1w EMA for primary trend, 1d Donchian for entry/exit with volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA50 on weekly close
    ema_50_1w = np.full(len(close_1w), np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_50_1w[i] = close_1w[i]
        elif np.isnan(ema_50_1w[i-1]):
            ema_50_1w[i] = close_1w[i]
        else:
            ema_50_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_50_1w[i-1]
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period) on daily data
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    for i in range(20, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i-20:i])
        donchian_low[i] = np.min(low_1d[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1w EMA50 to 6h
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Align 1d Donchian to 6h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_50 = ema_50_1w_aligned[i]
        dc_high = donchian_high_aligned[i]
        dc_low = donchian_low_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > 1w EMA50 + breaks above 1d Donchian high + volume confirmation
            if (price > ema_50 and price > dc_high and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < 1w EMA50 + breaks below 1d Donchian low + volume confirmation
            elif (price < ema_50 and price < dc_low and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low
            if price < dc_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high
            if price > dc_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_1d_Donchian_EMA_Volume"
timeframe = "6h"
leverage = 1.0