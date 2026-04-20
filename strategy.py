#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_Volume_1d_Trend_Filter
Hypothesis: Trade Keltner Channel breakouts on 12h with volume confirmation, filtered by 1d trend direction (EMA200).
Long when price breaks above upper KC with volume spike and 1d uptrend; short when breaks below lower KC with volume spike and 1d downtrend.
Uses volume spike (volume > 2.0x 20-period average) to confirm breakout strength.
Target: 60-120 total trades over 4 years (15-30/year) with position size 0.25 to balance opportunity and risk.
Works in bull/bear: 1d trend filter avoids counter-trend trades, volume confirmation reduces false breakouts.
"""

name = "12h_Keltner_Channel_Breakout_Volume_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema200_1d = ema(close_1d, 200)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate ATR(10) for Keltner Channel
    def calculate_atr(high_arr, low_arr, close_arr, period):
        tr = np.zeros_like(high_arr)
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(high_arr)):
            tr[i] = max(
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            )
        atr = np.zeros_like(high_arr)
        for i in range(len(atr)):
            if i < period:
                if i == 0:
                    atr[i] = tr[i]
                else:
                    atr[i] = (atr[i-1] * i + tr[i]) / (i + 1)
            else:
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 10)
    
    # Calculate EMA20 for Keltner Channel middle line
    ema20 = ema(close, 20)
    
    # Keltner Channel: Upper = EMA20 + 2*ATR, Lower = EMA20 - 2*ATR
    kc_upper = ema20 + 2 * atr
    kc_lower = ema20 - 2 * atr
    
    # Calculate volume spike (volume > 2.0x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper KC with volume spike AND 1d uptrend (price > EMA200)
            if close[i] > kc_upper[i] and volume_spike[i] and close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower KC with volume spike AND 1d downtrend (price < EMA200)
            elif close[i] < kc_lower[i] and volume_spike[i] and close[i] < ema200_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below lower KC OR 1d trend turns down
            if close[i] < kc_lower[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above upper KC OR 1d trend turns up
            if close[i] > kc_upper[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals