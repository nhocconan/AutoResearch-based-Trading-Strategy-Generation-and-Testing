#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) from 1d timeframe with EMA trend filter and volume confirmation.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13). 
# Enter long when Bull Power > 0 and increasing, price > EMA(34), and volume > 1.5x 20-bar average.
# Enter short when Bear Power < 0 and decreasing, price < EMA(34), and volume > 1.5x 20-bar average.
# Exit when power reverses or price crosses EMA(34).
# Elder Ray measures bull/bear strength relative to trend. Works in bull markets (strong Bull Power) and bear markets (strong Bear Power).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "6h_ElderRay_BullBearPower_EMA34_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA(13) for Elder Ray power calculation
    close_series_1d = pd.Series(close_1d)
    ema13_1d = close_series_1d.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA(13)
    bull_power = high_1d - ema13_1d
    
    # Bear Power = Low - EMA(13)
    bear_power = low_1d - ema13_1d
    
    # EMA(34) for trend filter
    ema34_1d = close_series_1d.ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align Elder Ray components and EMA34 to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray conditions
        bull_power_positive = bull_power_aligned[i] > 0
        bull_power_increasing = bull_power_aligned[i] > bull_power_aligned[i-1]
        bear_power_negative = bear_power_aligned[i] < 0
        bear_power_decreasing = bear_power_aligned[i] < bear_power_aligned[i-1]
        price_above_ema34 = close[i] > ema34_aligned[i]
        price_below_ema34 = close[i] < ema34_aligned[i]
        
        # Entry conditions
        long_entry = bull_power_positive and bull_power_increasing and price_above_ema34 and volume_confirm[i]
        short_entry = bear_power_negative and bear_power_decreasing and price_below_ema34 and volume_confirm[i]
        
        # Exit conditions: power reverses or price crosses EMA34
        long_exit = not (bull_power_positive and bull_power_increasing) or not price_above_ema34
        short_exit = not (bear_power_negative and bear_power_decreasing) or not price_below_ema34
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals