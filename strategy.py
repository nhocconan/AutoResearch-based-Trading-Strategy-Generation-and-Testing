#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Uses weekly Donchian channel to identify trend direction (bullish: price > weekly upper band; bearish: price < weekly lower band).
# Enters long when 12h price breaks above 12h upper Donchian band + volume + weekly bullish trend.
# Enters short when 12h price breaks below 12h lower Donchian band + volume + weekly bearish trend.
# Exits when price returns to the 12h middle band (20-period SMA of high-low).
# Designed to capture trends in both bull and bear markets by following weekly trend.
# Weekly trend filter reduces whipsaw and ensures alignment with higher timeframe momentum.
name = "12h_Donchian20_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian(20) for trend direction
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    weekly_upper = np.full_like(weekly_high, np.nan)
    weekly_lower = np.full_like(weekly_low, np.nan)
    
    for i in range(20, len(weekly_high)):
        weekly_upper[i] = np.max(weekly_high[i-20:i])
        weekly_lower[i] = np.min(weekly_low[i-20:i])
    
    weekly_upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    
    # 12h Donchian(20) for entry/exit
    high_12h = np.full_like(high, np.nan)
    low_12h = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        high_12h[i] = np.max(high[i-20:i])
        low_12h[i] = np.min(low[i-20:i])
    
    # 12h middle band (20-period SMA of (high+low)/2)
    median_price = (high + low) / 2
    middle_band = np.full_like(median_price, np.nan)
    for i in range(20, len(median_price)):
        middle_band[i] = np.mean(median_price[i-20:i])
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = np.full_like(volume, np.nan)
    vol_ema = 0.0
    for i in range(len(volume)):
        vol_ema = (volume[i] * 0.1) + (vol_ema * 0.9) if i >= 1 else volume[i]
        if i >= 19:  # approx 20-period EMA warmup
            vol_ema20[i] = vol_ema
    
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(middle_band[i]) or
            np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: 12h price breaks above upper band + volume + weekly bullish trend (price > weekly upper)
            if (price > high_12h[i] and vol_confirm[i] and price > weekly_upper_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h price breaks below lower band + volume + weekly bearish trend (price < weekly lower)
            elif (price < low_12h[i] and vol_confirm[i] and price < weekly_lower_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle band
            if price <= middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle band
            if price >= middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals