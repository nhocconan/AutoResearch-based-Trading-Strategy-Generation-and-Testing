#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume
Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 with volume spike and 1d uptrend bias.
Enter short when price breaks below Camarilla S1 with volume spike and 1d downtrend bias.
Exit when price reverts to Camarilla pivot or trend reverses.
Designed for 12h timeframe with 1d Camarilla levels and trend filter to limit trades to ~15-30/year.
Works in bull markets by buying strength at resistance breaks and in bear markets by selling weakness at support breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    # PP = (high + low + close) / 3
    diff = high_1d - low_1d
    R1 = close_1d + diff * 1.1 / 12
    S1 = close_1d - diff * 1.1 / 12
    PP = (high_1d + low_1d + close_1d) / 3
    
    # 1d EMA34 for trend filter
    ema34_1d = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        ema34_1d[34-1] = np.mean(close_1d[:34])
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] - ema34_1d[i-1]) * multiplier + ema34_1d[i-1]
    
    # Align all 1d indicators to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(PP_aligned[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: uptrend + break above R1 + volume
            if (price > ema34_1d_aligned[i] and  # 1d uptrend
                price > R1_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + break below S1 + volume
            elif (price < ema34_1d_aligned[i] and  # 1d downtrend
                  price < S1_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price back to pivot or trend reversal
            if price < PP_aligned[i] or price < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price back to pivot or trend reversal
            if price > PP_aligned[i] or price > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0