#!/usr/bin/env python3
"""
Hypothesis: 6-hour Williams %R (14) combined with weekly trend filter and volume confirmation.
Long when Williams %R crosses above -50 from oversold (< -80), weekly close > weekly SMA50 (uptrend), and volume > 1.5x average.
Short when Williams %R crosses below -50 from overbought (> -20), weekly close < weekly SMA50 (downtrend), and volume > 1.5x average.
Exit when Williams %R reverses across -50 or weekly trend changes.
Designed for low trade frequency (~15-35/year) to capture mean reversion within the weekly trend while avoiding counter-trend trades.
Works in both bull and bear markets by aligning with the weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMA50 for trend direction
    close_1w = df_1w['close'].values
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    weekly_uptrend = close_1w > sma50_1w  # True when above SMA50
    
    # Align weekly trend to 6-hour timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Calculate Williams %R (14-period) on 6-hour data
    highest_high = pd.Series(close).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(close).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume average (20-period) on 6-hour timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        weekly_up = weekly_uptrend_aligned[i] > 0.5  # Convert back to boolean
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Williams %R crosses above -50 from oversold (< -80), weekly uptrend, volume confirmation
            if (wr > -50 and wr_prev <= -50 and wr_prev < -80 and
                weekly_up and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -50 from overbought (> -20), weekly downtrend, volume confirmation
            elif (wr < -50 and wr_prev >= -50 and wr_prev > -20 and
                  not weekly_up and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses back below -50 OR weekly trend turns down
                if (wr < -50 and wr_prev >= -50) or not weekly_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses back above -50 OR weekly trend turns up
                if (wr > -50 and wr_prev <= -50) or weekly_up:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0