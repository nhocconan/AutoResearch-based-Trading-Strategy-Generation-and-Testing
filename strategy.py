#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA trend filter. Long when Bull Power > 0 and price above 1d EMA50, short when Bear Power < 0 and price below 1d EMA50. Uses discrete sizing (0.25) to minimize fees. Designed for 12-30 trades/year, works in bull markets via trend-following longs and in bear markets via trend-following shorts when aligned with higher timeframe trend.
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
    
    # Get 1d data for HTF EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components on 6h
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and price above 1d EMA50
            long_signal = (bull_power[i] > 0) and (close[i] > ema_50_aligned[i])
            # Short: Bear Power < 0 and price below 1d EMA50
            short_signal = (bear_power[i] < 0) and (close[i] < ema_50_aligned[i])
            
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
            # Exit when Bull Power <= 0 (momentum weakness)
            exit_signal = bull_power[i] <= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power >= 0 (momentum weakness)
            exit_signal = bear_power[i] >= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0