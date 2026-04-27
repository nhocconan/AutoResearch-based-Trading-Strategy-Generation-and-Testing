#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MA_Crossover_v1
Hypothesis: Elder Ray (Bull/Bear Power) combined with Zero-Lag Moving Average crossover on 6h timeframe, filtered by 1-week trend. 
Elder Ray identifies institutional buying/selling pressure. Zero-Lag MA reduces lag for timely entries. 
Weekly trend filter ensures we trade with higher timeframe momentum. 
This combination should work in both bull and bear markets by capturing momentum shifts with institutional confirmation.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Zero-Lag Moving Average (ZLMA) on 6h data
    # ZLMA = 2*EMA - EMA(EMA) to reduce lag
    ema1 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema2 = pd.Series(ema1).ewm(span=21, adjust=False, min_periods=21).mean().values
    zlma = 2 * ema1 - ema2
    
    # Calculate EMA13 for crossover signal
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray (Bull Power and Bear Power) using 13-period EMA
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_close  # Bull Power = High - EMA13
    bear_power = low - ema13_close   # Bear Power = Low - EMA13
    
    # Weekly trend filter: price vs 50-period EMA on weekly
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align all 6h indicators (they're already LTF, but ensure proper indexing)
    # For Elder Ray and ZLMA, we need to align to ensure no look-ahead
    # Since these are calculated from close prices, we'll use them directly but ensure warmup
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level to reduce fee churn)
    
    # Warmup: need EMA21 (21), EMA13 (13), weekly EMA50 (50)
    start_idx = max(21, 13, 50)
    
    for i in range(start_idx, n):
        # Skip if weekly trend data not ready
        if np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        zlma_val = zlma[i]
        ema13_val = ema13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        weekly_ema50 = ema50_1w_aligned[i]
        
        # Determine weekly trend
        weekly_uptrend = close_val > weekly_ema50
        weekly_downtrend = close_val < weekly_ema50
        
        if position == 0:
            # Long entry: Bull Power > 0 (buying pressure) + ZLMA crosses above EMA13 + weekly uptrend
            if (bull_power_val > 0 and 
                zlma_val > ema13_val and 
                zlma[i-1] <= ema13[i-1] and  # crossover confirmation
                weekly_uptrend):
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short entry: Bear Power < 0 (selling pressure) + ZLMA crosses below EMA13 + weekly downtrend
            elif (bear_power_val < 0 and 
                  zlma_val < ema13_val and 
                  zlma[i-1] >= ema13[i-1] and  # crossover confirmation
                  weekly_downtrend):
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: ZLMA crosses below EMA13 OR Bear Power turns negative (selling pressure)
            if (zlma_val < ema13_val and zlma[i-1] >= ema13[i-1]) or bear_power_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: ZLMA crosses above EMA13 OR Bull Power turns positive (buying pressure)
            if (zlma_val > ema13_val and zlma[i-1] <= ema13[i-1]) or bull_power_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MA_Crossover_v1"
timeframe = "6h"
leverage = 1.0