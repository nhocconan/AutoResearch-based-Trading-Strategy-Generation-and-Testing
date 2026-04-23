#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
- Long: Close breaks above 20-period high + price > 1d EMA34 (uptrend) + volume > 1.8x 20-period average
- Short: Close breaks below 20-period low + price < 1d EMA34 (downtrend) + volume > 1.8x 20-period average
- Exit: Close retreats below midline (10-period average of high/low) for longs, or above midline for shorts
- Uses Donchian channels for structure, 1d EMA for trend filter, volume spike for confirmation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation)
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Highest high and lowest low over last 20 periods
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Midline for exit: average of 20-period high and low
    midline = (high_ma + low_ma) / 2.0
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(midline[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above 20-period high + uptrend + volume spike
        # Short: Close breaks below 20-period low + downtrend + volume spike
        long_signal = (close[i] > high_ma[i] and 
                      uptrend and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < low_ma[i] and 
                       downtrend and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retreats below midline (for longs) or above midline (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: Close moves back below midline
                if close[i] <= midline[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close moves back above midline
                if close[i] >= midline[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0