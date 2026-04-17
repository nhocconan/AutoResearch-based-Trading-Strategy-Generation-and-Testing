#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_TrendFilter_v2
Strategy: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
Long: Price breaks above R1 with volume > 1.5x 20-period average and 1d EMA34 up
Short: Price breaks below S1 with volume > 1.5x 20-period average and 1d EMA34 down
Exit: Price returns to Pivot point or volume drops below average
Position size: 0.25
Designed to capture institutional breakout attempts with trend and volume confirmation.
Timeframe: 12h
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
    
    # Calculate 1d EMA34 for trend filter (using daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for volume confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot levels for previous day
        # Need previous day's high, low, close
        if i < 2:  # Need at least 2 periods for previous day data
            signals[i] = 0.0
            continue
            
        # Get previous day's OHLC (assuming 2 periods per day for 12h timeframe)
        prev_high = high[i-2]
        prev_low = low[i-2]
        prev_close = close[i-2]
        
        # Calculate pivot point and support/resistance levels
        pivot = (prev_high + prev_low + prev_close) / 3.0
        r1 = pivot + (prev_high - prev_low) * 1.1 / 12.0
        s1 = pivot - (prev_high - prev_low) * 1.1 / 12.0
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Entry conditions
        if position == 0:
            # Long: Price breaks above R1 with volume and 1d uptrend
            if (close[i] > r1 and 
                volume_filter and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):  # 1d EMA34 rising
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume and 1d downtrend
            elif (close[i] < s1 and 
                  volume_filter and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):  # 1d EMA34 falling
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price returns to pivot or volume drops
            if close[i] <= pivot or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price returns to pivot or volume drops
            if close[i] >= pivot or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_TrendFilter_v2"
timeframe = "12h"
leverage = 1.0