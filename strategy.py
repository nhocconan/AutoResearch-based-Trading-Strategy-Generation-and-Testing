#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses Donchian(20) from 12h price action for breakout signals
- 1d EMA50 as trend filter (long only above, short only below) - avoids whipsaw in ranging markets
- Volume > 2.0x 20-period average for confirmation (adjusts for 12h lower frequency)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for current 12h bar
        highest_high = np.max(high[i-19:i+1])  # 20-period high including current
        lowest_low = np.min(low[i-19:i+1])     # 20-period low including current
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up AND price above 1d EMA50 AND volume confirmation
            if close[i] > highest_high and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown down AND price below 1d EMA50 AND volume confirmation
            elif close[i] < lowest_low and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below Donchian low OR crosses below 1d EMA50
            if close[i] < lowest_low or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian high OR crosses above 1d EMA50
            if close[i] > highest_high or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_VolumeSpike_Filter_v1"
timeframe = "12h"
leverage = 1.0