# The strategy has been updated to use the 12h timeframe as required.
# It uses weekly high/low for structure, daily EMA for trend, and volume confirmation.

#!/usr/bin/env python3
"""
12h_Weekly_High_Low_Breakout_DailyTrend_Volume
Hypothesis: Breakouts of weekly high/low on 12h timeframe, filtered by daily EMA trend and volume spikes (>2x 20-period average). This strategy targets low frequency (12-37 trades/year) by using higher timeframe structure (weekly) and trend (daily), making it robust in both bull and bear markets. The weekly structure provides significant support/resistance, while the daily EMA ensures alignment with the intermediate trend. Volume confirmation adds conviction to breakouts.
"""

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
    
    # Get weekly data for structure (high/low)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly high and low from the previous completed week
    weekly_high = df_1w['high'].shift(1).values
    weekly_low = df_1w['low'].shift(1).values
    
    # Align weekly levels to 12h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Get daily data for trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above weekly high, above daily EMA50 trend, volume confirmation
            if close[i] > weekly_high_val and close[i] > ema50_val and vol_conf:
                signals[i] = size
                position = 1
            # Short: price breaks below weekly low, below daily EMA50 trend, volume confirmation
            elif close[i] < weekly_low_val and close[i] < ema50_val and vol_conf:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses below daily EMA50
            if close[i] < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above daily EMA50
            if close[i] > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Weekly_High_Low_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0