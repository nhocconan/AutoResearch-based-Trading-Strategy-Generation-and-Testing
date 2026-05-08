#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_Reversal_Confirmation"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Bull Power and Bear Power aligned to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 60-period SMA for trend filter on 6h
    close_s = pd.Series(close)
    sma60 = close_s.rolling(window=60, min_periods=60).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(sma60[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bear Power > 0 (sellers exhausted) AND price above SMA60 (uptrend)
            long_cond = (bear_power_aligned[i] > 0) and (close[i] > sma60[i])
            
            # Short: Bull Power < 0 (buyers exhausted) AND price below SMA60 (downtrend)
            short_cond = (bull_power_aligned[i] < 0) and (close[i] < sma60[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power < 0 (buyers exhausted) OR price crosses below SMA60
            if (bull_power_aligned[i] < 0) or (close[i] < sma60[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power > 0 (sellers exhausted) OR price crosses above SMA60
            if (bear_power_aligned[i] > 0) or (close[i] > sma60[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray measures bull/bear power relative to EMA13. 
# Long when bear power turns positive (sellers exhausted) in uptrend (price > SMA60).
# Short when bull power turns negative (buyers exhausted) in downtrend (price < SMA60).
# Works in bull markets (buy dips via bull power exhaustion) and bear markets (sell rallies via bear power exhaustion).
# Uses 60-period SMA for trend alignment to avoid counter-trend whipsaws.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.