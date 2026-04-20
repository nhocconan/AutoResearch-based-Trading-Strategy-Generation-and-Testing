#!/usr/bin/env python3
"""
6h_1d_ElderRay_With1dTrendFilter_v1
Concept: Elder Ray Index (Bull/Bear Power) with daily EMA trend filter for 6h timeframe.
- Long when Bull Power > 0 (close > EMA13) AND daily trend is up (price > daily EMA50)
- Short when Bear Power < 0 (close < EMA13) AND daily trend is down (price < daily EMA50)
- Exit when power crosses zero or daily trend changes
- Uses 13-period EMA for power calculation (standard Elder Ray)
- Conservative sizing (0.25) to manage drawdown in volatile 6h bars
- Works in bull (trend following) and bear (counter-trend when daily trend aligns)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_With1dTrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 6h: Calculate EMA13 for Elder Ray power calculation ===
    close = prices['close'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Close - EMA13
    bull_power = close - ema13
    # Bear Power = EMA13 - Close (negative when close > EMA13)
    bear_power = ema13 - close
    
    # === Daily: Calculate EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA calculations
    
    for i in range(start_idx, n):
        # Get values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        close_val = close[i]
        ema50_1d_val = ema50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(close_val) or np.isnan(ema50_1d_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND price above daily EMA50 (uptrend)
            if bull_val > 0 and close_val > ema50_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND price below daily EMA50 (downtrend)
            elif bear_val > 0 and close_val < ema50_1d_val:  # bear_val > 0 means close < EMA13
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price breaks below daily EMA50
            if bull_val <= 0 or close_val < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns negative OR price breaks above daily EMA50
            if bear_val <= 0 or close_val > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals