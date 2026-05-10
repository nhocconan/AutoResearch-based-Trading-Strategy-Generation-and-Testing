#!/usr/bin/env python3
"""
6H_LiquiditySweep_1dTrend_Confirmation
Hypothesis: During 6h sessions, liquidity sweeps (taking out recent highs/lows) followed by reversal in direction of 1d trend capture smart money reversals. Works in both bull/bear markets by aligning with higher timeframe trend. Low trade frequency via requirement of liquidity sweep + trend confirmation + volume spike.
"""

name = "6H_LiquiditySweep_1dTrend_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA 50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2.0x 24-period average (4-day average on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 2.0
    
    # Liquidity sweep detection: take out recent 6-period high/low then reverse
    # For longs: take out low of last 6 bars, then close above that low
    # For shorts: take out high of last 6 bars, then close below that high
    lookback = 6
    
    # Calculate rolling min/max for liquidity levels
    roll_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 50, 24)  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or np.isnan(roll_min[i]) or np.isnan(roll_max[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Check for liquidity sweep and reversal
        bullish_setup = (low[i] <= roll_min[i-1] and  # took out recent low
                         close[i] > roll_min[i-1] and  # closed back above it (reversal)
                         volume[i] > vol_threshold[i] and
                         is_uptrend)
        
        bearish_setup = (high[i] >= roll_max[i-1] and  # took out recent high
                         close[i] < roll_max[i-1] and  # closed back below it (reversal)
                         volume[i] > vol_threshold[i] and
                         is_downtrend)
        
        if position == 0:
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on trend reversal or opposite liquidity sweep
            if not is_uptrend or (high[i] >= roll_max[i-1] and close[i] < roll_max[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on trend reversal or opposite liquidity sweep
            if not is_downtrend or (low[i] <= roll_min[i-1] and close[i] > roll_min[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals