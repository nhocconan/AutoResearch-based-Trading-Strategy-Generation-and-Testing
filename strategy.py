#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Go long when Bull Power > 0 and Bear Power < 0 with weekly uptrend and volume spike
# Go short when Bear Power < 0 and Bull Power < 0 with weekly downtrend and volume spike
# Designed for low trade frequency in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_ElderRay_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def ema(data, period):
    """Exponential Moving Average with proper handling"""
    return pd.Series(data).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate weekly EMA(13) for trend direction
    close_1w = df_1w['close'].values
    ema13_1w = ema(close_1w, 13)
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Elder Ray components on 6h data
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13_1w_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema13_1w_val = ema13_1w_aligned[i]
        ema13_val = ema13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Bull Power > 0, Bear Power < 0, weekly uptrend, volume spike
            if (bull_val > 0 and bear_val < 0 and 
                close[i] > ema13_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0, Bull Power < 0, weekly downtrend, volume spike
            elif (bear_val < 0 and bull_val < 0 and 
                  close[i] < ema13_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR weekly trend breaks
            if not (bull_val > 0 and bear_val < 0) or close[i] < ema13_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Bull Power >= 0 OR weekly trend breaks
            if not (bull_val < 0 and bear_val < 0) or close[i] > ema13_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals