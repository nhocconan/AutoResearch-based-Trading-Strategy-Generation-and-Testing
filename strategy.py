#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
# Uses weekly pivot levels from previous week to determine institutional bias.
# Long when price breaks above Donchian high AND price > weekly pivot (bullish bias).
# Short when price breaks below Donchian low AND price < weekly pivot (bearish bias).
# Volume confirmation (>2x 20-period average) ensures breakout strength.
# Designed for low trade frequency (~15-30/year) to minimize fee decay.
# Weekly pivot provides multi-timeframe structure that works in both bull and bear markets
# by aligning with higher timeframe institutional levels.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (using previous week's data)
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    
    # Calculate 20-period Donchian channels on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high: highest high over past 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over past 20 periods
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_1w[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pivot_val = pivot_1w[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + bullish weekly bias + volume spike
            if price > dc_high and price > pivot_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + bearish weekly bias + volume spike
            elif price < dc_low and price < pivot_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below Donchian low or weekly pivot
                if price < dc_low or price < pivot_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above Donchian high or weekly pivot
                if price > dc_high or price > pivot_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0