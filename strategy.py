#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrendFilter
Hypothesis: Elder Ray Bull/Bear Power combined with 1d EMA50 trend filter on 6h timeframe.
Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and Bear Power rising in uptrend.
Short when Bear Power < 0 and Bull Power falling in downtrend. Uses discrete sizing (0.25) to minimize fees.
Works in bull markets (buying strength) and bear markets (selling weakness). Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Calculate EMA13 on 6h close for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 and EMA50
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Elder Ray signals in direction of 1d trend
            # Long: Bull Power > 0 (buying strength) AND Bear Power rising (increasing strength) in uptrend
            # Short: Bear Power > 0 (selling pressure) AND Bull Power falling (decreasing strength) in downtrend
            long_signal = (bull_power[i] > 0) and (bear_power[i] > bear_power[i-1]) and (close[i] > ema50_aligned[i])
            short_signal = (bear_power[i] > 0) and (bull_power[i] < bull_power[i-1]) and (close[i] < ema50_aligned[i])
            
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
            # Exit when Bull Power turns negative (buying strength gone)
            exit_signal = bull_power[i] <= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power turns negative (selling pressure gone)
            exit_signal = bear_power[i] <= 0
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrendFilter"
timeframe = "6h"
leverage = 1.0