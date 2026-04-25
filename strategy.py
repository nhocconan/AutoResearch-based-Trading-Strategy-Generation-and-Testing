#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_12hTrend_Filter
Hypothesis: Elder Ray Bull/Bear Power on 6h with 12h EMA34 trend filter.
Bull Power = High - EMA13, Bear Power = Low - EMA13.
Enter long when Bull Power > 0 and 12h trend up (close > EMA34).
Enter short when Bear Power < 0 and 12h trend down (close < EMA34).
Exit when power reverses sign or trend fails.
Uses discrete sizing (0.25) for low trade frequency (~15-25/year) to work in both bull and bear markets.
Elder Ray measures price strength relative to EMA, filtering weak moves in chop.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h close for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # Calculate EMA13 on 6h for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13_6h  # High - EMA13
    bear_power = low - ema13_6h   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA13 (13) and EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(ema13_6h[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for entry signals with trend filter
            # Long: Bull Power > 0 and 12h trend up (close > EMA34)
            # Short: Bear Power < 0 and 12h trend down (close < EMA34)
            long_signal = (bull_power[i] > 0) and (close[i] > ema34_12h_aligned[i])
            short_signal = (bear_power[i] < 0) and (close[i] < ema34_12h_aligned[i])
            
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
            # Exit when Bull Power <= 0 (weakening strength) OR trend fails
            exit_signal = (bull_power[i] <= 0) or (close[i] < ema34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when Bear Power >= 0 (weakening strength) OR trend fails
            exit_signal = (bear_power[i] >= 0) or (close[i] > ema34_12h_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_12hTrend_Filter"
timeframe = "6h"
leverage = 1.0