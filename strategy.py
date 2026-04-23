#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme with 1d EMA34 Trend Filter and Volume Spike
- Long: Williams %R < -80 (oversold) + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Williams %R > -20 (overbought) + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Williams %R crosses above -50 for longs exit, below -50 for shorts exit
- Uses Williams %R for momentum extremes, 1d EMA34 for trend filter, volume spike for confirmation
- Works in both bull and bear markets by capturing mean reversion in trends
- Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
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
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Williams %R needs 14, EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Williams %R signals with trend filter and volume confirmation
        # Long: Williams %R < -80 (oversold) + uptrend + volume spike
        # Short: Williams %R > -20 (overbought) + downtrend + volume spike
        long_signal = (williams_r[i] < -80 and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (momentum fading)
                if williams_r[i] >= -50:
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R crosses below -50 (momentum fading)
                if williams_r[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0