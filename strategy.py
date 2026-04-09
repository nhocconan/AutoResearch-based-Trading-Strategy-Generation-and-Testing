#!/usr/bin/env python3
# 6h_1d_price_position_reversal_v1
# Hypothesis: On 6-hour chart, mean reversion when price reaches extreme daily price position levels.
# Uses daily price position (PP = (Close - Low)/(High - Low)) to identify overbought/oversold conditions.
# Extreme PP (<0.2 or >0.8) combined with 6h RSI extremes signals mean reversion.
# Works in both bull and bear markets as it captures overextended moves regardless of trend.
# Target: 60-120 total trades over 4 years (15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_price_position_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily price position: (Close - Low)/(High - Low)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_range = daily_high - daily_low
    # Avoid division by zero
    daily_range = np.where(daily_range == 0, 1e-10, daily_range)
    daily_pp = (daily_close - daily_low) / daily_range
    
    # Align daily price position to 6h timeframe
    daily_pp_aligned = align_htf_to_ltf(prices, df_1d, daily_pp)
    
    # Calculate 6h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if any required data is invalid
        if np.isnan(daily_pp_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to neutral PP or RSI > 50
            if daily_pp_aligned[i] >= 0.5 or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to neutral PP or RSI < 50
            if daily_pp_aligned[i] <= 0.5 or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: oversold conditions (PP < 0.2 and RSI < 30)
            if daily_pp_aligned[i] < 0.2 and rsi[i] < 30:
                position = 1
                signals[i] = 0.25
            # Enter short: overbought conditions (PP > 0.8 and RSI > 70)
            elif daily_pp_aligned[i] > 0.8 and rsi[i] > 70:
                position = -1
                signals[i] = -0.25
    
    return signals