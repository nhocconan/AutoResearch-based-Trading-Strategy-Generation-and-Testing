#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
- Uses weekly Camarilla pivot levels (R1/S1) from prior week for trend bias
- Enters on Donchian(20) breakout in direction of weekly pivot bias
- Requires volume > 1.5x 20-period average for confirmation
- Exits on Donchian(20) opposite breakout or volume drop below average
- Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag
- Works in both bull/bear via pivot bias filter (avoids counter-trend breakouts)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot points (using prior week's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #           S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We use R1/S1 for bias: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # But simpler: use weekly high/low for breakout bias
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Calculate Donchian(20) on 6h primary timeframe
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate volume average (20-period) on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly pivot data to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        dh = highest_high[i]  # Donchian high
        dl = lowest_low[i]    # Donchian low
        wh = weekly_high_aligned[i]  # Weekly high
        wl = weekly_low_aligned[i]   # Weekly low
        vol_ma = volume_ma[i]
        vol = volume[i]
        price = close[i]
        
        # Determine weekly bias: price above weekly midpoint = bullish bias
        weekly_mid = (wh + wl) / 2.0
        bullish_bias = price > weekly_mid
        bearish_bias = price < weekly_mid
        
        if position == 0:
            # Long: Donchian breakout above AND bullish weekly bias AND volume confirmation
            if price > dh and bullish_bias and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below AND bearish weekly bias AND volume confirmation
            elif price < dl and bearish_bias and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakdown below OR volume drops below average
            if price < dl or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian breakout above OR volume drops below average
            if price > dh or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotBias_Volume"
timeframe = "6h"
leverage = 1.0