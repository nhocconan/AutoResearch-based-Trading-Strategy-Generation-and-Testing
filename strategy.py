#!/usr/bin/env python3
"""
1d_1w_Weekly_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v1
Hypothesis: On daily timeframe, use weekly Camarilla pivot levels (R1/S1) as breakout triggers.
Enter long when price breaks above weekly R1 with volume confirmation and price > weekly EMA20 (uptrend filter).
Enter short when price breaks below weekly S1 with volume confirmation and price < weekly EMA20 (downtrend filter).
Exit on opposite breakout or trend reversal.
Designed for 1d timeframe to target 20-40 trades/year with high-conviction entries.
Works in bull markets by capturing continuation breaks and in bear markets by capturing breakdowns.
Weekly timeframe provides stable structure; daily execution improves timing.
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
    """Calculate Camarilla pivot levels for given period"""
    # Typical price
    tp = (high + low + close) / 3.0
    # Range
    r = high - low
    # Camarilla levels
    S1 = tp - (1.1 * r / 12)
    S2 = tp - (1.1 * r / 6)
    S3 = tp - (1.1 * r / 4)
    R1 = tp + (1.1 * r / 12)
    R2 = tp + (1.1 * r / 6)
    R3 = tp + (1.1 * r / 4)
    return R1, R2, R3, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for Camarilla, EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA20 for trend filter
    ema20_1w = calculate_ema(close_1w, 20)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly Camarilla levels (recalculate each week)
    R1_1w = np.full_like(close_1w, np.nan)
    S1_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(df_1w)):
        R1, _, _, S1, _, _ = calculate_camarilla(high_1w[i], low_1w[i], close_1w[i])
        R1_1w[i] = R1
        S1_1w[i] = S1
    
    R1_1w_aligned = align_htf_to_ltf(prices, df_1w, R1_1w)
    S1_1w_aligned = align_htf_to_ltf(prices, df_1w, S1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(R1_1w_aligned[i]) or 
            np.isnan(S1_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = prices['volume'].iloc[i] > 1.3 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Uptrend: price > weekly EMA20
            if price > ema20_1w_aligned[i]:
                # Long: price breaks above weekly R1 with volume confirmation
                if price > R1_1w_aligned[i] and volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < weekly EMA20
            elif price < ema20_1w_aligned[i]:
                # Short: price breaks below weekly S1 with volume confirmation
                if price < S1_1w_aligned[i] and volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: trend reversal or opposite breakout
            if price < ema20_1w_aligned[i] or price < S1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or opposite breakout
            if price > ema20_1w_aligned[i] or price > R1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Weekly_Camarilla_R1S1_Breakout_Volume_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0