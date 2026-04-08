#!/usr/bin/env python3
"""
1d Bollinger Band Breakout with Weekly Volume Confirmation
Hypothesis: Price breaks above/below Bollinger Bands on daily chart when weekly 
volume is above average, capturing momentum in both bull and bear markets. 
Weekly volume filter avoids false breakouts during low-volume periods.
Target: 10-25 trades/year per symbol.
"""

name = "1d_bollinger_breakout_weekly_volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for volume confirmation - call ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    volume_weekly = df_weekly['volume'].values
    
    # Calculate Bollinger Bands (20-period, 2 std)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Calculate weekly volume average (10-period)
    vol_ma_10_weekly = pd.Series(volume_weekly).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_10_weekly[i]):
            signals[i] = 0.0
            continue
        
        # Get aligned weekly volume MA for current daily bar
        vol_ma_10_weekly_aligned = align_htf_to_ltf(prices, df_weekly, vol_ma_10_weekly)[i]
        
        # Volume confirmation: current weekly volume above 10-period average
        volume_confirm = volume_weekly[i] > vol_ma_10_weekly_aligned
        
        if position == 1:  # Long position
            # Exit on middle band (mean reversion)
            if close[i] < sma_20[i]:
                position = 0
                signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit on middle band (mean reversion)
            if close[i] > sma_20[i]:
                position = 0
                signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry logic: Bollinger Band breakout with weekly volume confirmation
            if volume_confirm:
                if close[i] > upper_band[i] and close[i-1] <= upper_band[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < lower_band[i] and close[i-1] >= lower_band[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals