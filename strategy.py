#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_Filter
Hypothesis: Elder Ray (Bull/Bear Power) on 6h with 1-week EMA50 trend filter. 
Only trade in direction of weekly trend: long when Bull Power > 0 in uptrend, 
short when Bear Power < 0 in downtrend. Uses discrete sizing (0.25) to minimize fee churn.
Designed for low trade frequency (~10-20/year) to work in both bull and bear markets.
Elder Ray measures buying/selling pressure relative to EMA13, providing early momentum signals.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 on 6h for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power (Elder Ray)
    bull_power = high - ema13  # Buying pressure: high minus EMA13
    bear_power = low - ema13   # Selling pressure: low minus EMA13
    
    # Align HTF EMA50 to 6h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and EMA50 (50)
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Elder Ray signals with weekly trend filter
            # Long: Bull Power > 0 (buying pressure) in uptrend (close > weekly EMA50)
            # Short: Bear Power < 0 (selling pressure) in downtrend (close < weekly EMA50)
            long_signal = (bull_power[i] > 0) and (close[i] > ema50_aligned[i])
            short_signal = (bear_power[i] < 0) and (close[i] < ema50_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when Bull Power turns negative (loss of buying pressure)
            exit_signal = bull_power[i] <= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power turns positive (loss of selling pressure)
            exit_signal = bear_power[i] >= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0