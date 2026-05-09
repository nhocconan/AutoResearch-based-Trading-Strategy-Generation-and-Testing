#!/usr/bin/env python3
"""
6h_ERP_Energy_Trend
Elder Ray Power (Bull/Bear Power) with 12h trend filter and volume confirmation.
- Bull Power = High - EMA13
- Bear Power = Low - EMA13
Long when Bull Power > 0 and rising, Bear Power < 0, and 12h trend up.
Short when Bear Power < 0 and falling, Bull Power > 0, and 12h trend down.
Volume filter ensures institutional participation.
Designed for 6h timeframe to capture trends while avoiding whipsaws.
Works in both bull and bear markets via trend filter.
Target: 20-50 trades/year.
"""

name = "6h_ERP_Energy_Trend"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    ema34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema34_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema34_12h[i] = (close_12h[i] * 2 + ema34_12h[i-1] * 32) / 34
    
    # Align 12h EMA34 to 6h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate EMA13 for Elder Ray Power on 6h data
    ema13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[0:13])
        for i in range(13, len(close)):
            ema13[i] = (close[i] * 2 + ema13[i-1] * 11) / 13
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 20)  # Need 12h EMA34, EMA13, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema34_12h_aligned[i]
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        volume_surge = volume_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: Uptrend + Bull Power rising > 0 + Bear Power < 0 + volume surge
            if trend_up and bull_power_rising and bull_power[i] > 0 and bear_power[i] < 0 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + Bear Power falling < 0 + Bull Power > 0 + volume surge
            elif not trend_up and bear_power_falling and bear_power[i] < 0 and bull_power[i] > 0 and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR Bull Power turns negative OR Bear Power turns positive
            if not trend_up or bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR Bear Power turns positive OR Bull Power turns negative
            if trend_up or bear_power[i] >= 0 or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals