#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1w Choppiness Index filter to avoid choppy markets.
# Long: Price crosses above 1d EMA200 + 1w Choppiness > 61.8 (range) + volume > 1.5x average volume.
# Short: Price crosses below 1d EMA200 + 1w Choppiness > 61.8 (range) + volume > 1.5x average volume.
# Uses 1d EMA200 for trend, 1w Choppiness for regime filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        alpha = 2 / (200 + 1)
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # 1w data for Choppiness Index
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Choppiness Index (14-period)
    chop_1w = np.full(len(close_1w), np.nan)
    atr_1w = np.full(len(close_1w), np.nan)
    
    # Calculate ATR for Choppiness
    tr_1w = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], 
                       abs(high_1w[i] - close_1w[i-1]), 
                       abs(low_1w[i] - close_1w[i-1]))
    
    # Calculate ATR(14)
    for i in range(14, len(close_1w)):
        if i == 14:
            atr_1w[i] = np.nanmean(tr_1w[1:15])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Calculate Choppiness Index
    for i in range(14, len(close_1w)):
        atr_sum = np.nansum(atr_1w[i-13:i+1])
        highest_high = np.nanmax(high_1w[i-13:i+1])
        lowest_low = np.nanmin(low_1w[i-13:i+1])
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop_1w[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA200 to 12h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Align 1w Choppiness to 12h (needs extra 2-bar delay for confirmation)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema200 = ema200_1d_aligned[i]
        chop = chop_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Choppiness filter: only trade in ranging markets (Choppiness > 61.8)
        range_filter = chop > 61.8
        
        if position == 0:
            # Long: price crosses above EMA200 + range + volume confirmation
            if (price > ema200 and range_filter and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price crosses below EMA200 + range + volume confirmation
            elif (price < ema200 and range_filter and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA200
            if price < ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above EMA200
            if price > ema200:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_EMA200_1w_Chop_Volume"
timeframe = "12h"
leverage = 1.0