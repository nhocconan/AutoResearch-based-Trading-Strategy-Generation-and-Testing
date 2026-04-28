#!/usr/bin/env python3
"""
4h_48h_Range_With_Volume_And_Trend
Hypothesis: Uses 48-hour (2-day) price range to identify accumulation/distribution zones with volume confirmation and 12h EMA trend filter. Trades breakouts from the 48h range in the direction of the 12h trend. Designed for low trade frequency (15-30/year) to minimize fee decay while capturing strong directional moves in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 48h high/low for range (2 days of 4h data)
    # We need to look back 12 periods of 4h data for 48h range
    high_48h = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_48h = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    # Volume confirmation: >1.8x 48-period MA (8 days of 4h bars)
    vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and 48h range to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(high_48h[i]) or
            np.isnan(low_48h[i]) or
            np.isnan(vol_ma_48[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (>1.8x average)
        vol_confirm = volume[i] > (1.8 * vol_ma_48[i])
        
        # Breakout conditions from 48h range
        long_breakout = close[i] > high_48h[i] and vol_confirm and uptrend
        short_breakout = close[i] < low_48h[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of 48h range
        midpoint_48h = (high_48h[i] + low_48h[i]) / 2
        long_exit = close[i] < midpoint_48h
        short_exit = close[i] > midpoint_48h
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
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
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_48h_Range_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0