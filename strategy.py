#!/usr/bin/env python3
"""
6H_Price_Action_Breakout_with_1D_Trend_and_Volume
Hypothesis: 6h price breaks above/below the prior 6h high/low with 1D EMA50 trend confirmation and volume spike.
This captures momentum continuation in both bull and bear markets by filtering breakouts with higher timeframe trend.
Volume confirmation reduces false breakouts. Targets 12-37 trades/year to minimize fee drag on 6h timeframe.
"""
name = "6H_Price_Action_Breakout_with_1D_Trend_and_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1D EMA50 for trend direction
    close_1d_series = pd.Series(df_1d['close'])
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter: current 6h volume > 2.0 x 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    # Calculate prior 6h high/low for breakout levels
    # Shift by 1 to use only completed bars
    prior_high = np.roll(high, 1)
    prior_low = np.roll(low, 1)
    prior_high[0] = np.nan  # First bar has no prior
    prior_low[0] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(50, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(prior_high[i]) or np.isnan(prior_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 24 bars between trades (4 days on 6h TF) to reduce frequency
            if bars_since_exit < 24:
                continue
                
            # Long: price breaks above prior 6h high with 1D EMA50 uptrend and volume spike
            if (close[i] > prior_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: price breaks below prior 6h low with 1D EMA50 downtrend and volume spike
            elif (close[i] < prior_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: price returns to opposite EMA50 side (trend reversal)
            if position == 1 and close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals