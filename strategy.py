#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v2
Hypothesis: Use 1w Camarilla pivot levels (R1/S1) as breakout levels on 1d.
Enter long when price breaks above weekly R1 with volume confirmation and 1w uptrend.
Enter short when price breaks below weekly S1 with volume confirmation and 1w downtrend.
Exit when price returns to weekly pivot (PP) or trend reverses.
Designed for 1d timeframe with 1w filters to limit trades to ~10-20/year.
Works in bull markets by buying strength and in bear markets by selling weakness.
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_ = high - low
    pp = (high + low + close) / 3
    r1 = close + range_ * 1.1 / 12
    s1 = close - range_ * 1.1 / 12
    return pp, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data once for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA10 for trend filter
    ema10_1w = calculate_ema(close_1w, 10)
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # 1w Camarilla levels (calculate from prior week's OHLC)
    pp_1w = np.full_like(high_1w, np.nan)
    r1_1w = np.full_like(high_1w, np.nan)
    s1_1w = np.full_like(low_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        pp_1w[i], r1_1w[i], s1_1w[i] = calculate_camarilla(
            high_1w[i-1], low_1w[i-1], close_1w[i-1]
        )
    
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema10_1w_aligned[i]) or np.isnan(pp_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: uptrend + break above weekly R1 + volume
            if (price > ema10_1w_aligned[i] and 
                price > r1_1w_aligned[i] and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: downtrend + break below weekly S1 + volume
            elif (price < ema10_1w_aligned[i] and 
                  price < s1_1w_aligned[i] and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to weekly PP or trend reversal
            if price < pp_1w_aligned[i] or price < ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to weekly PP or trend reversal
            if price > pp_1w_aligned[i] or price > ema10_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v2"
timeframe = "1d"
leverage = 1.0