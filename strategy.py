#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Volume
Camarilla pivot breakout with volume confirmation:
- Long when price breaks above R1 (1.0833 level) + volume > 1.5x 20-period avg
- Short when price breaks below S1 (0.9167 level) + volume > 1.5x 20-period avg
- Exit on opposite S1/R1 break
- Uses 1w trend filter: only long when price > 1w EMA34, only short when price < 1w EMA34
- Designed for 15-25 trades/year per symbol
Works in both bull (captures breakouts) and bear (short breakdowns) markets
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
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    if len(close_1w) >= 34:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1w = np.full(len(close_1w), np.nan)
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_12h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 12h bar
    # R1 = close + (high - low) * 1.0833
    # S1 = close - (high - low) * 1.0833
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        if not (np.isnan(high[i-1]) or np.isnan(low[i-1]) or np.isnan(close[i-1])):
            r1[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.0833
            s1[i] = close[i-1] - (high[i-1] - low[i-1]) * 0.9167  # Note: 1 - 1.0833 = -0.0833, but standard is 0.9167
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # need sufficient data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + above 1w EMA34
            if close[i] > r1[i] and vol_filter and close[i] > ema_34_1w_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + below 1w EMA34
            elif close[i] < s1[i] and vol_filter and close[i] < ema_34_1w_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reverse to short)
            if close[i] < s1[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reverse to long)
            if close[i] > r1[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume"
timeframe = "12h"
leverage = 1.0