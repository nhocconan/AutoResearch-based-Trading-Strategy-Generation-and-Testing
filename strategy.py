#!/usr/bin/env python3
"""
1h_RangeBreakout_4hTrend_Volume
Hypothesis: Trade 1h breakouts in direction of 4h EMA trend, confirmed by volume >1.5x average.
Uses 4h EMA(34) as trend filter to avoid counter-trend trades. Range breakouts capture momentum
after volatility expansion. Position size 0.20 to limit drawdown. Target 15-37 trades/year.
Works in both bull/bear markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_period = 34
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period-1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * 2 / (ema_period + 1)) + (ema_4h[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 4h EMA to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h range breakout parameters
    lookback = 20
    breakout_mult = 1.5
    
    # Calculate rolling high/low for breakout levels
    roll_high = np.full_like(high, np.nan)
    roll_low = np.full_like(low, np.nan)
    for i in range(lookback, n):
        roll_high[i] = np.max(high[i-lookback:i])
        roll_low[i] = np.min(low[i-lookback:i])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, vol_period, ema_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above rolling high with volume and above 4h EMA
            if close[i] > roll_high[i] and vol_confirm and close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below rolling low with volume and below 4h EMA
            elif close[i] < roll_low[i] and vol_confirm and close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price closes below rolling low or below 4h EMA
            if close[i] < roll_low[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price closes above rolling high or above 4h EMA
            if close[i] > roll_high[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RangeBreakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0