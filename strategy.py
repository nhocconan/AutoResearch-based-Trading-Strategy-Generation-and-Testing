#!/usr/bin/env python3
# 6h_ElderRay_RayBand_Breakout
# Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure. 
# Combines with 1d trend (EMA34) and volume confirmation to filter false signals.
# Works in bull markets by capturing strong bullish power breakouts and in bear markets 
# by capturing bearish power breakdowns. Uses Elder Ray bands (EMA13 of Bull/Bear Power) 
# as dynamic support/resistance for breakout entries.

name = "6h_ElderRay_RayBand_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate daily EMA13 for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = (df_1d['high'].values - ema13_1d)
    bear_power = (df_1d['low'].values - ema13_1d)
    
    # Calculate Elder Ray Bands (EMA13 of Bull/Bear Power)
    bull_ema13 = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_ema13 = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align Elder Ray bands to 6t
    bull_ema13_aligned = align_htf_to_ltf(prices, df_1d, bull_ema13)
    bear_ema13_aligned = align_htf_to_ltf(prices, df_1d, bear_ema13)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation (20-period MA on 6h = ~5 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 (13) and volume MA (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_ema13_aligned[i]) or 
            np.isnan(bear_ema13_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation (>1.5x MA to balance sensitivity and filtering)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: bullish power above its EMA13 + uptrend + volume
            if bull_power[i] > bull_ema13_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish power below its EMA13 + downtrend + volume
            elif bear_power[i] < bear_ema13_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bullish power crosses below EMA13 or trend breaks
            if bull_power[i] < bull_ema13_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bearish power crosses above EMA13 or trend breaks
            if bear_power[i] > bear_ema13_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals