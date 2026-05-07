#!/usr/bin/env python3
# 6h_LiquiditySweep_Reversal_1wTrend_Volume
# Hypothesis: 6h chart strategy that identifies liquidity sweeps (false breakouts) at weekly high/low levels and reverses with the weekly trend.
# In ranging markets, price often sweeps liquidity pools before reversing. In trending markets, we only take reversals in the direction of the weekly trend.
# Volume confirmation ensures the sweep had participation. Designed to work in both bull and bear markets by filtering with weekly trend.
# Target: 15-35 trades/year per symbol to minimize fee drag while capturing high-probability reversals.

timeframe = "6h"
name = "6h_LiquiditySweep_Reversal_1wTrend_Volume"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and liquidity levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate weekly high and low for liquidity levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume spike detection: 2x average volume (12-period = 3 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 12)  # Ensure we have EMA50 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price sweeps below weekly low (liquidity grab) then closes back above it with volume, in uptrend
            if (low[i] < weekly_low_aligned[i] and 
                close[i] > weekly_low_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price sweeps above weekly high (liquidity grab) then closes back below it with volume, in downtrend
            elif (high[i] > weekly_high_aligned[i] and 
                  close[i] < weekly_high_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below weekly low (trend failure)
            if close[i] < weekly_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above weekly high (trend failure)
            if close[i] > weekly_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals