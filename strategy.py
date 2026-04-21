#!/usr/bin/env python3
"""
1d_1W_PriceChannel_Breakout_VolumeFilter
Hypothesis: On the daily timeframe, price breaks of a 20-period Donchian channel with volume confirmation capture strong momentum moves. This works in both bull and bear markets because breakouts signal the start of new trends, and the volume filter ensures only high-conviction moves are taken. Weekly trend filter (price above/below 40-week EMA) avoids counter-trend trades. Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter (40-week EMA)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 40:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    # Calculate 40-week EMA
    ema_weekly = np.zeros_like(close_weekly)
    for i in range(len(close_weekly)):
        if i < 40:
            ema_weekly[i] = np.mean(close_weekly[:i+1])
        else:
            ema_weekly[i] = close_weekly[i] * 0.0488 + ema_weekly[i-1] * (1 - 0.0488)  # alpha = 2/(40+1)
    
    # Align weekly EMA to daily
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Load daily data for Donchian channel (we are on 1d timeframe, so prices is already daily)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period) on daily data
    upper = np.zeros_like(high)
    lower = np.zeros_like(low)
    for i in range(len(high)):
        if i < 20:
            upper[i] = np.max(high[:i+1])
            lower[i] = np.min(low[:i+1])
        else:
            upper[i] = np.max(high[i-20:i+1])
            lower[i] = np.min(low[i-20:i+1])
    
    # Volume filter: current volume > 1.5x 20-day average
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i+1])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(ema_weekly_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        up = upper[i]
        low_ch = lower[i]
        vol_ok = volume_filter[i]
        weekly_ema = ema_weekly_aligned[i]
        
        # Exit conditions
        if position == 1:
            # Exit long if price breaks below lower band or weekly trend turns down
            if price < low_ch or price < weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
            continue
        elif position == -1:
            # Exit short if price breaks above upper band or weekly trend turns up
            if price > up or price > weekly_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
            continue
        
        # Entry conditions (only when flat)
        if position == 0:
            # Long: price breaks above upper Donchian band with volume confirmation and weekly uptrend
            if price > up and vol_ok and price > weekly_ema:
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian band with volume confirmation and weekly downtrend
            elif price < low_ch and vol_ok and price < weekly_ema:
                signals[i] = -0.30
                position = -1
                entry_price = price
    
    return signals

name = "1d_1W_PriceChannel_Breakout_VolumeFilter"
timeframe = "1d"
leverage = 1.0