#!/usr/bin/env python3
"""
4h_Donchian20_12hTrend_Filter_VolumeSpike
Hypothesis: Breakout above Donchian(20) high or below Donchian(20) low on 4h, filtered by 12-hour EMA trend and volume spike. Works in both bull and bear by trading with the 12h trend. Targets 20-40 trades/year to minimize fee drag.
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
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) on 4h data
    # Highest high of last 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 bars
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike (>2.0x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12-hour EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Price relative to Donchian levels
        price_above_upper = close[i] > highest_high[i]
        price_below_lower = close[i] < lowest_low[i]
        
        # Entry logic:
        # Long: Breakout above Donchian high in uptrend
        long_entry = vol_confirm and trend_up and price_above_upper and (close[i-1] <= highest_high[i-1])
        
        # Short: Breakdown below Donchian low in downtrend
        short_entry = vol_confirm and trend_down and price_below_lower and (close[i-1] >= lowest_low[i-1])
        
        # Exit logic: Opposite break or trend reversal
        long_exit = price_below_lower or not trend_up
        short_exit = price_above_upper or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
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
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_12hTrend_Filter_VolumeSpike"
timeframe = "4h"
leverage = 1.0