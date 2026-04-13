#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h price action combined with 1-day Bollinger Band width regime filter.
# Strategy uses Bollinger Band width to identify low volatility (squeeze) conditions on daily timeframe,
# then enters on 4h breakouts of the 20-period Donchian channel in the direction of the 4h EMA(50) trend.
# Bollinger Band width < 0.05 indicates volatility contraction, often preceding expansion and trend continuation.
# This filters out false breakouts during high volatility choppy periods.
# Position size: 0.25 (25%) to manage drawdown risk.
# Target: 20-50 trades per year (80-200 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Bollinger Band width (volatility regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Bollinger Bands (20, 2)
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 19:
            continue
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Align Bollinger Band width to 4h timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # 4h Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            continue
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 4h EMA(50) for trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema_multiplier = 2 / (50 + 1)
        ema50[49] = np.mean(close[:50])  # Simple average for first value
        for i in range(50, n):
            ema50[i] = (close[i] - ema50[i-1]) * ema_multiplier + ema50[i-1]
    
    # Volume confirmation: current volume > 1.5x 20-period average
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        bb_width_val = bb_width_aligned[i]
        
        # Volatility regime filter: Bollinger Band width < 0.05 indicates low volatility (squeeze)
        low_volatility = bb_width_val < 0.05
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price breaks above Donchian upper band + above EMA50 + low volatility + volume confirmation
            if (price > highest_high[i] and
                price > ema50[i] and
                low_volatility and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below Donchian lower band + below EMA50 + low volatility + volume confirmation
            elif (price < lowest_low[i] and
                  price < ema50[i] and
                  low_volatility and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price breaks below Donchian lower band or below EMA50
            if (price < lowest_low[i] or price < ema50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price breaks above Donchian upper band or above EMA50
            if (price > highest_high[i] or price > ema50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BB_Width_Donchian_Breakout"
timeframe = "4h"
leverage = 1.0