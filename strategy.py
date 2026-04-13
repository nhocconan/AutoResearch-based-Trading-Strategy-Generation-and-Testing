#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d EMA trend filter and 4h Donchian breakout.
# Long: Price breaks above Donchian(20) high + EMA50(1d) > EMA200(1d) + volume > 1.5x average volume (20-period).
# Short: Price breaks below Donchian(20) low + EMA50(1d) < EMA200(1d) + volume > 1.5x average volume.
# Uses 1d EMA crossover for trend filter, 4h Donchian breakout for entry with volume confirmation.
# Trend filter ensures we trade in the direction of higher timeframe trend.
# Volume confirmation adds confirmation of institutional interest.
# Position size: 0.25 (25% of capital) to manage risk during drawdowns.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 and EMA200 on daily timeframe
    ema50_1d = np.full(len(close_1d), np.nan)
    ema200_1d = np.full(len(close_1d), np.nan)
    
    # EMA50 calculation
    alpha_50 = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
        elif np.isnan(ema50_1d[i-1]):
            ema50_1d[i] = close_1d[i]
        else:
            ema50_1d[i] = alpha_50 * close_1d[i] + (1 - alpha_50) * ema50_1d[i-1]
    
    # EMA200 calculation
    alpha_200 = 2.0 / (200 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema200_1d[i] = close_1d[i]
        elif np.isnan(ema200_1d[i-1]):
            ema200_1d[i] = close_1d[i]
        else:
            ema200_1d[i] = alpha_200 * close_1d[i] + (1 - alpha_200) * ema200_1d[i-1]
    
    # Donchian channel (20-period) on 4h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d EMA values to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        ema50 = ema50_1d_aligned[i]
        ema200 = ema200_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # Trend filter: EMA50 > EMA200 for uptrend, EMA50 < EMA200 for downtrend
        uptrend = ema50 > ema200
        downtrend = ema50 < ema200
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume confirmation
            if (price > d_high and uptrend and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + downtrend + volume confirmation
            elif (price < d_low and downtrend and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (opposite boundary)
            if price < d_low:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (opposite boundary)
            if price > d_high:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA_Trend_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0