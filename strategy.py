#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily high-low range breakout on 12h timeframe with volume confirmation.
# Uses the previous day's high-low range as a volatility-based breakout level.
# Enters long when price breaks above previous day's high + 0.2 * daily range,
# and short when price breaks below previous day's low - 0.2 * daily range.
# Includes volume confirmation (volume > 1.5x 20-period MA) to avoid false breakouts.
# Designed for low trade frequency (target 20-50/year) to minimize fee drag in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for high, low, and close
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range (high - low)
    daily_range = high_1d - low_1d
    
    # Calculate breakout levels: previous day's high + 0.2*range, low - 0.2*range
    # Use shift(1) to avoid look-ahead (use previous day's data)
    breakout_high = np.roll(high_1d, 1) + 0.2 * np.roll(daily_range, 1)
    breakout_low = np.roll(low_1d, 1) - 0.2 * np.roll(daily_range, 1)
    # First day has no previous day
    breakout_high[0] = np.nan
    breakout_low[0] = np.nan
    
    # Align daily breakout levels to 12h timeframe
    breakout_high_aligned = align_htf_to_ltf(prices, df_1d, breakout_high)
    breakout_low_aligned = align_htf_to_ltf(prices, df_1d, breakout_low)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(breakout_high_aligned[i]) or np.isnan(breakout_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above previous day's high + 0.2*range with volume
            if close[i] > breakout_high_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below previous day's low - 0.2*range with volume
            elif close[i] < breakout_low_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price breaks below previous day's low - 0.2*range
            if close[i] < breakout_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above previous day's high + 0.2*range
            if close[i] > breakout_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyRangeBreakout_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0