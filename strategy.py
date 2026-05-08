#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power via EMA(13): Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Go long when Bull Power > 0 and Bear Power < 0 (bullish market structure) + 1d EMA(34) uptrend + volume spike.
# Go short when Bear Power < 0 and Bull Power < 0 (bearish market structure) + 1d EMA(34) downtrend + volume spike.
# Designed to capture strong directional moves in both bull and bear markets with low trade frequency.

name = "6h_ElderRay_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def ema(series, period):
    """Exponential Moving Average"""
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend direction
    close_1d = df_1d['close'].values
    ema34_1d = ema(close_1d, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Elder Ray components on 6h data
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 AND Bear Power < 0 (bullish structure) + uptrend + volume spike
            if (bull_val > 0 and bear_val < 0 and 
                close[i] > ema34_1d_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 AND Bull Power < 0 (bearish structure) + downtrend + volume spike
            elif (bear_val < 0 and bull_val < 0 and 
                  close[i] < ema34_1d_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bullish structure breaks OR price breaks below trend
            if not (bull_val > 0 and bear_val < 0) or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bearish structure breaks OR price breaks above trend
            if not (bear_val < 0 and bull_val < 0) or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals