#!/usr/bin/env python3
"""
12h_ema_trend_1w_volume_v1
Hypothesis: On 12-hour timeframe, use EMA trend from weekly timeframe with volume confirmation. 
Enter long when price > weekly EMA20 with volume > 1.5x average, short when price < weekly EMA20 with volume > 1.5x average. 
Exit when price crosses back over weekly EMA20. Weekly EMA provides strong trend filter that works in both bull/bear markets. 
Designed for low frequency (12-37 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_trend_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Align to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume average warmup
        # Skip if weekly data not available
        if np.isnan(ema_20_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price crosses below weekly EMA20
            if close[i] <= ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above weekly EMA20
            if close[i] >= ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above weekly EMA20 with volume confirmation
            long_entry = (close[i] > ema_20_aligned[i]) and vol_confirm
            # Short entry: price below weekly EMA20 with volume confirmation
            short_entry = (close[i] < ema_20_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals