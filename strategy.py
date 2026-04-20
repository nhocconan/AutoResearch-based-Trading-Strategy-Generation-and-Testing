#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Trend Filter
# - Elder Ray (Bull/Bear Power) on 6h: Bull = High - EMA13, Bear = EMA13 - Low
# - Long when Bull Power > 0 and Bear Power < 0 AND 12h EMA34 > 12h EMA89 (uptrend)
# - Short when Bear Power > 0 and Bull Power < 0 AND 12h EMA34 < 12h EMA89 (downtrend)
# - Exit when Elder Ray signals weaken or 12h trend flips
# - Combines momentum (Elder Ray) with trend filter (12h EMA cross) for robust signals
# - Designed for 6h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA8 and EMA34 on 12h timeframe
    ema8_12h = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMAs to 6h timeframe
    ema8_12h_aligned = align_htf_to_ltf(prices, df_12h, ema8_12h)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Elder Ray on 6h timeframe
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # EMA13 for Elder Ray calculation
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = EMA13 - Low
    bear_power = ema13_6h - low_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in indicators
        if (np.isnan(ema8_12h_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema8 = ema8_12h_aligned[i]
        ema34 = ema34_12h_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        if position == 0:
            # Long entry: Bull Power > 0, Bear Power < 0, and 12h EMA8 > EMA34 (uptrend)
            if bull > 0 and bear < 0 and ema8 > ema34:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0, Bull Power < 0, and 12h EMA8 < EMA34 (downtrend)
            elif bear > 0 and bull < 0 and ema8 < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Elder Ray weakens or 12h trend turns down
            if bull <= 0 or bear >= 0 or ema8 <= ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Elder Ray weakens or 12h trend turns up
            if bear <= 0 or bull >= 0 or ema8 >= ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_12hEMATrendFilter"
timeframe = "6h"
leverage = 1.0