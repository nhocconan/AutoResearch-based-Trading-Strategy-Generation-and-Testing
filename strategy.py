#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_BearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter (weekly EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 100:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray calculations (daily EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily EMA13 for Elder Ray calculations
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Need enough data for weekly EMA200
    
    for i in range(start_idx, n):
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_up = ema200_1w_aligned[i]  # Weekly trend filter
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (bullish momentum) and price above weekly EMA200
            if bull_val > 0 and close[i] > trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (bearish momentum) and price below weekly EMA200
            elif bear_val < 0 and close[i] < trend_up:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative (momentum fading)
            if bull_val <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive (momentum fading)
            if bear_val >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals