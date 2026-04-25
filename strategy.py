#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Filter
Hypothesis: Elder Ray Bull/Bear Power on 6h timeframe with 1d EMA50 trend filter. 
Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and price > 1d EMA50 (uptrend). 
Short when Bear Power < 0 and price < 1d EMA50 (downtrend). Uses discrete sizing (0.25) to minimize fees.
Elder Ray measures buying/selling pressure relative to short-term trend; combining with higher-timeframe trend 
filters reduces false signals in chop. Works in bull markets via long entries and bear markets via short entries 
when aligned with daily trend. Target: 15-25 trades/year (60-100 total) for low fee drag.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 on 6h for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure
    
    # Align HTF EMA50 to 6h timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and EMA50 (50)
    start_idx = max(13, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Elder Ray signals with trend filter
            # Long: Bull Power > 0 (buying pressure) and price > 1d EMA50 (uptrend)
            # Short: Bear Power < 0 (selling pressure) and price < 1d EMA50 (downtrend)
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
            # Exit when Bull Power turns negative OR price breaks below EMA50
            exit_signal = (bull_power[i] <= 0) or (close[i] < ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power turns positive OR price breaks above EMA50
            exit_signal = (bear_power[i] >= 0) or (close[i] > ema50_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0