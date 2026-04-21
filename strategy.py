#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_Volume
Hypothesis: Use 12h Camarilla pivot levels (R1/S1) as structure. Breakout above R1 with volume = long, breakdown below S1 with volume = short. 12h trend filter (price > 12h EMA25) ensures alignment with higher timeframe momentum. Designed for 4h timeframe with 12h filters to limit trades to ~20-50/year. Works in bull markets by buying strength and in bear markets by selling weakness, using volume to confirm breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for pivot levels and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA25 for trend filter
    ema25_12h = calculate_ema(close_12h, 25)
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Calculate 12h Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    r1_12h = close_12h + (high_12h - low_12h) * 1.1 / 12.0
    s1_12h = close_12h - (high_12h - low_12h) * 1.1 / 12.0
    
    # Align pivot levels to 4h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema25_12h_aligned[i]) or np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i])):
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
            if (price > ema25_12h_aligned[i] and 
                price > r1_12h_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + break below S1 + volume
            elif (price < ema25_12h_aligned[i] and 
                  price < s1_12h_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or price below S1 (reversal signal)
            if price < ema25_12h_aligned[i] or price < s1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or price above R1 (reversal signal)
            if price > ema25_12h_aligned[i] or price > r1_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0