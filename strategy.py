#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h price breaking above/below 1d high/low with volume confirmation.
# Uses 1-day high/low as support/resistance levels, volume surge for breakout strength,
# and avoids false breakouts with volume filter. Designed for low-frequency, high-conviction trades.
# Target: 15-30 trades/year to minimize fee drag on 12h timeframe.
name = "12h_DailyHighLow_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily high/low levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous day's high and low (completed day only)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Align 1d high/low to 12h timeframe (previous day's levels)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume confirmation: volume > 1.5x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for volume EMA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price breaks above previous day's high + volume confirmation
            if price > daily_high_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below previous day's low + volume confirmation
            elif price < daily_low_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below previous day's low or volume dries up
            if price < daily_low_aligned[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above previous day's high or volume dries up
            if price > daily_high_aligned[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals