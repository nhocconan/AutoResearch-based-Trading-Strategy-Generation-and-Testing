#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with weekly trend filter and volume spike
# Long when price breaks above Donchian(20) high + close > weekly EMA50 (uptrend) + volume spike
# Short when price breaks below Donchian(20) low + close < weekly EMA50 (downtrend) + volume spike
# Exit when price crosses Donchian middle (mean of 20-period high/low) or trend reverses
# Uses weekly EMA50 to filter trend direction, reducing whipsaw in sideways markets.
# Designed for low trade frequency (~15-30/year) on 12h timeframe to minimize fee drain.
# Works in both bull (breakouts with trend) and bear (breakdowns with trend) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-period EMA on weekly close for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian(20): 20-period high and low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        highest = highest_high[i]
        lowest = lowest_low[i]
        mid = donchian_mid[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: break above Donchian high + uptrend + volume spike
            if price > highest and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below Donchian low + downtrend + volume spike
            elif price < lowest and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses Donchian middle or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below Donchian middle or trend turns down
                if price < mid or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above Donchian middle or trend turns up
                if price > mid or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0