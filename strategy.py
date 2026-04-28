#!/usr/bin/env python3
"""
12h_Trend_Breakout_Volume_Confirmation
Hypothesis: On 12h timeframe, use 1d trend (via 34-period EMA) as filter for Donchian(20) breakouts with volume confirmation (>1.5x 20-period average volume). This captures strong trend continuation moves while avoiding counter-trend trades. Volume surge confirms institutional participation. Designed for low trade frequency (~20-30/year) to minimize fee decay and work in both bull/bear markets via trend alignment.
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
    
    # Get daily data for trend filter and Donchian calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily 34 EMA for trend filter
    close_daily = df_daily['close'].values
    ema34_daily = pd.Series(close_daily).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily EMA to 12h timeframe
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Calculate Donchian channels (20-period) on 12h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for daily EMA34 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_daily_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Donchian breakout with trend filter and volume surge
        long_breakout = close[i] > high_20[i-1]  # Break above prior 20-period high
        short_breakout = close[i] < low_20[i-1]  # Break below prior 20-period low
        daily_uptrend = ema34_daily_aligned[i] > close_daily[0]  # Above EMA34 (trend up)
        daily_downtrend = ema34_daily_aligned[i] < close_daily[0]  # Below EMA34 (trend down)
        
        long_entry = long_breakout and daily_uptrend and volume_surge[i]
        short_entry = short_breakout and daily_downtrend and volume_surge[i]
        
        # Exit on opposite breakout with volume surge
        long_exit = short_breakout and volume_surge[i]
        short_exit = long_breakout and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Trend_Breakout_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0