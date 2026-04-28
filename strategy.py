#!/usr/bin/env python3
"""
1h_Turtle_Soup_4hTrend_VolumeFilter
Hypothesis: 1-hour Turtle Soup strategy with 4-hour trend filter and volume confirmation.
Goes long when price breaks above recent low with bullish divergence, short when breaks below recent high with bearish divergence.
Uses 4-hour EMA for trend direction and volume spike (>1.5x 20-period MA) for confirmation.
Designed for low trade frequency (15-25 trades/year) to minimize fee drag while capturing reversal opportunities.
Works in both bull and bear markets by following 4h trend direction and using mean-reversion entries.
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h 20-period highest high and lowest low for entry levels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Turtle Soup entry conditions
        # Long: price breaks above recent low (stop hunt) then reverses up
        long_setup = low[i] <= low_min_20[i] and close[i] > low_min_20[i]
        # Short: price breaks below recent high (stop hunt) then reverses down
        short_setup = high[i] >= high_max_20[i] and close[i] < high_max_20[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: turtle soup in direction of trend with volume
        long_entry = vol_confirm and uptrend and long_setup
        short_entry = vol_confirm and downtrend and short_setup
        
        # Exit logic: opposite setup or trend change
        long_exit = short_setup or (not uptrend)
        short_exit = long_setup or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Turtle_Soup_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0