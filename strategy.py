#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d EMA trend filter and volume confirmation
# Uses bull power (high - EMA) and bear power (EMA - low) to measure bull/bear strength.
# Enters long when bull power > 0 and increasing + price > 1d EMA50 + volume > average.
# Enters short when bear power > 0 and increasing + price < 1d EMA50 + volume > average.
# Uses EMA13 on 6h for power calculation. Avoids whipsaws by requiring power to be rising.
# Target: 60-120 total trades over 4 years (15-30/year) with trend-following edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for price action and Elder Ray calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 6h for Elder Ray power
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    bull_power = high_6h - ema13_6h  # High minus EMA
    bear_power = ema13_6h - low_6h   # EMA minus Low
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period on 6h)
    vol_avg_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_6h)
    
    # Calculate rising power (current > previous)
    bull_power_rising = bull_power_aligned > np.roll(bull_power_aligned, 1)
    bear_power_rising = bear_power_aligned > np.roll(bear_power_aligned, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: bull power > 0 and rising + price > 1d EMA50 + volume > average
        if (bull_power_aligned[i] > 0 and bull_power_rising[i] and
            close[i] > ema50_1d_aligned[i] and
            volume[i] > vol_avg_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: bear power > 0 and rising + price < 1d EMA50 + volume > average
        elif (bear_power_aligned[i] > 0 and bear_power_rising[i] and
              close[i] < ema50_1d_aligned[i] and
              volume[i] > vol_avg_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or power becomes negative
        elif position == 1 and (bull_power_aligned[i] <= 0 or not bull_power_rising[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] <= 0 or not bear_power_rising[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_Power_Trend"
timeframe = "6h"
leverage = 1.0