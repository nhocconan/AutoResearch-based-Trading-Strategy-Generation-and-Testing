#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 1-day Elder Ray power + trend filter.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Bullish when Bull Power > 0 and rising, Bearish when Bear Power > 0 and rising.
# Use 1-day EMA(13) for Elder Ray, aligned to 6h.
# Entry: Long when Bull Power > 0 and rising + price > EMA(34) trend filter.
# Short when Bear Power > 0 and rising + price < EMA(34) trend filter.
# Exit when power turns negative or trend fails.
# Designed to work in both bull/bear by capturing institutional buying/selling pressure.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-day data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # 1-day EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Elder Ray components
    bull_power = high_1d - ema_13_1d
    bear_power = ema_13_1d - low_1d
    
    # Align Elder Ray components
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate Elder Ray slope (change from previous day)
    bull_power_slope = np.diff(bull_power_aligned, prepend=bull_power_aligned[0])
    bear_power_slope = np.diff(bear_power_aligned, prepend=bear_power_aligned[0])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough for EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(bull_power_slope[i]) or np.isnan(bear_power_slope[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1-day EMA(34)
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Elder Ray conditions: power positive AND rising
        bull_strong = bull_power_aligned[i] > 0 and bull_power_slope[i] > 0
        bear_strong = bear_power_aligned[i] > 0 and bear_power_slope[i] > 0
        
        # Entry conditions
        long_entry = bull_strong and uptrend
        short_entry = bear_strong and downtrend
        
        # Exit conditions: power turns negative or trend fails
        if position == 1:
            exit_condition = not bull_strong or not uptrend
        elif position == -1:
            exit_condition = not bear_strong or not downtrend
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "6h_ElderRay_Power_Trend"
timeframe = "6h"
leverage = 1.0