#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Turtle Soup strategy with 1-week trend filter and volume confirmation.
# Turtle Soup is a reversal pattern that fades false breakouts.
# In a weekly uptrend, we look for failed new highs (long setup).
# In a weekly downtrend, we look for failed new lows (short setup).
# Combined with volume confirmation to filter low-probability signals.
# Target: 15-35 trades per year (60-140 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA(50) for weekly trend filter
    ema50_1w = np.zeros(len(close_1w))
    ema_multiplier = 2 / (50 + 1)
    ema50_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema50_1w[i] = (close_1w[i] - ema50_1w[i-1]) * ema_multiplier + ema50_1w[i-1]
    
    # Align weekly EMA to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period highest high and lowest low for Turtle Soup setup
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        weekly_trend = ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long setup: Weekly uptrend + price makes new 20-period high but closes below it (failed breakout)
            if (price > weekly_trend and
                high[i] > highest_high[i] and  # made new high
                close[i] < highest_high[i] and  # but closed below it (failed breakout)
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short setup: Weekly downtrend + price makes new 20-period low but closes above it (failed breakdown)
            elif (price < weekly_trend and
                  low[i] < lowest_low[i] and   # made new low
                  close[i] > lowest_low[i] and  # but closed above it (failed breakdown)
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price closes above the failed breakout level (invalidates setup) or weekly trend turns down
            if (close[i] > highest_high[i] or
                price < weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price closes below the failed breakdown level (invalidates setup) or weekly trend turns up
            if (close[i] < lowest_low[i] or
                price > weekly_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w_TurtleSoup_Reversal_Volume"
timeframe = "12h"
leverage = 1.0