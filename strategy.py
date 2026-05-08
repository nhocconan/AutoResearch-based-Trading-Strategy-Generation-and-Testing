#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with weekly trend filter and volume confirmation
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to detect institutional buying/selling pressure.
# We go long when bull power crosses above zero with weekly EMA(40) uptrend and volume spike.
# We go short when bear power crosses below zero with weekly EMA(40) downtrend and volume spike.
# Uses 6h timeframe to target 12-37 trades/year, avoiding excessive frequency.
# Elder Ray provides early signals of institutional accumulation/distribution.
# Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume spike confirms institutional participation in the move.

name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA(40) for trend filter
    weekly_close = df_1w['close'].values
    ema40_1w = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate EMA(13) for Elder Ray (standard period)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema40_1w_val = ema40_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume spike: current volume > 2.0 * 20-period average on 6h
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume[i] > (2.0 * vol_ma[i]) if not np.isnan(vol_ma[i]) else False
        
        if position == 0:
            # Enter long: bull power crosses above zero + weekly uptrend + volume spike
            if (bull_val > 0 and bull_power[i-1] <= 0 and 
                close[i] > ema40_1w_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: bear power crosses below zero + weekly downtrend + volume spike
            elif (bear_val < 0 and bear_power[i-1] >= 0 and 
                  close[i] < ema40_1w_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bear power crosses below zero OR weekly trend turns down
            if (bear_val < 0 and bear_power[i-1] >= 0) or close[i] < ema40_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bull power crosses above zero OR weekly trend turns up
            if (bull_val > 0 and bull_power[i-1] <= 0) or close[i] > ema40_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals