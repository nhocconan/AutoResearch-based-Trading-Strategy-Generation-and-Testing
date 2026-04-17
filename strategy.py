#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above Donchian upper band + volume > 1.5x 20-period average + price > 12h EMA34.
Short when price breaks below Donchian lower band + volume > 1.5x 20-period average + price < 12h EMA34.
Exit on opposite Donchian break or volume dry-up.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Donchian channels (20-period)
    upper_channel = np.zeros(n)
    lower_channel = np.zeros(n)
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate volume average (20-period)
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(34, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_channel[i]
        breakout_down = close[i] < lower_channel[i]
        
        # Trend filter from 12h EMA34
        price = close[i]
        ema34_val = ema34_12h_aligned[i]
        uptrend = price > ema34_val
        downtrend = price < ema34_val
        
        if position == 0:
            # Long: Donchian upper breakout + volume confirmation + uptrend
            if breakout_up and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout + volume confirmation + downtrend
            elif breakout_down and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Donchian lower break OR volume dry-up (< 0.5x average)
            if breakout_down or volume[i] < 0.5 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Donchian upper break OR volume dry-up
            if breakout_up or volume[i] < 0.5 * vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_12hEMA34_Trend"
timeframe = "4h"
leverage = 1.0