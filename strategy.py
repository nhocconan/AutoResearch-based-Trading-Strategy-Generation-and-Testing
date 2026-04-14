#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 1-day Elder Ray (Bull/Bear Power) with regime filter.
Long when Bull Power > 0 and Bear Power < 0 + price above EMA13 (bullish regime).
Short when Bear Power > 0 and Bull Power < 0 + price below EMA13 (bearish regime).
Exit when power signals reverse or price crosses EMA13.
Elder Ray measures bull/bear strength relative to EMA13, effective in both trending and ranging markets.
Designed for low turnover: ~15-25 trades/year per symbol to minimize fee drift.
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
    
    # Load 1-day data once for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on 1-day close
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high_1d - ema13  # High minus EMA13
    bear_power = low_1d - ema13   # Low minus EMA13
    
    # Align 1-day indicators to 6h timeframe (using previous values to avoid look-ahead)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(30, n):  # Start after warmup period
        # Get aligned values for current bar
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        ema13_val = ema13_aligned[i]
        
        if np.isnan(bull) or np.isnan(bear) or np.isnan(ema13_val):
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying pressure) AND Bear Power < 0 (no selling pressure)
            # Plus price above EMA13 for bullish regime confirmation
            if bull > 0 and bear < 0 and close[i] > ema13_val:
                position = 1
                signals[i] = position_size
            # Short: Bear Power > 0 (strong selling pressure) AND Bull Power < 0 (no buying pressure)
            # Plus price below EMA13 for bearish regime confirmation
            elif bear > 0 and bull < 0 and close[i] < ema13_val:
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit long: Bull Power turns negative OR price crosses below EMA13
            if bull <= 0 or close[i] < ema13_val:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: Bear Power turns negative OR price crosses above EMA13
            if bear <= 0 or close[i] > ema13_val:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "6h_1d_ElderRay_EMA13_Regime"
timeframe = "6h"
leverage = 1.0