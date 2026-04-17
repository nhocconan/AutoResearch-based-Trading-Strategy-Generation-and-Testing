#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index with 1d Regime Filter.
Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
Long when Bull Power > 0 AND Bear Power increasing (less negative) AND 1d close > 1d EMA50 (bullish regime).
Short when Bear Power < 0 AND Bull Power decreasing (less positive) AND 1d close < 1d EMA50 (bearish regime).
Exit when power signals diverge or regime flips.
Uses 13-period EMA for sensitivity to 6h momentum shifts, 1d EMA50 for regime filter to avoid counter-trend trades.
Target: 80-180 total trades over 4 years (20-45/year). Elder Ray captures momentum exhaustion, 
regime filter ensures alignment with higher-timeframe trend to reduce whipsaw in ranging markets.
"""

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
    
    # Calculate 13-period EMA for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Higher highs vs trend
    bear_power = low - ema13   # Lower lows vs trend
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        bp = bull_power[i]
        br = bear_power[i]
        ema50 = ema50_1d_aligned[i]
        price = close[i]
        
        if position == 0:
            # Long: Bull Power > 0 (strong highs) AND Bear Power increasing (less negative) AND bullish regime
            if i > start_idx:
                bp_prev = bull_power[i-1]
                br_prev = bear_power[i-1]
                bull_increasing = bp > bp_prev
                bear_increasing = br > br_prev  # less negative = increasing
                if bp > 0 and bear_increasing and price > ema50:
                    signals[i] = 0.25
                    position = 1
            # Short: Bear Power < 0 (strong lows) AND Bull Power decreasing (less positive) AND bearish regime
            elif i > start_idx:
                bp_prev = bull_power[i-1]
                br_prev = bear_power[i-1]
                bull_decreasing = bp < bp_prev  # less positive = decreasing
                bear_decreasing = br < br_prev  # more negative = decreasing
                if br < 0 and bull_decreasing and price < ema50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns negative OR Bull Power turns negative OR regime flip
            if br < 0 or bp < 0 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive OR Bear Power turns positive OR regime flip
            if bp > 0 or br > 0 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Regime"
timeframe = "6h"
leverage = 1.0