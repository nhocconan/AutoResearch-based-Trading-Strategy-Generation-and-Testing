#!/usr/bin/env python3
"""
6h_Premium_Index_Divergence_12hTrend
Hypothesis: Use futures premium index (basis) as a sentiment indicator. When basis diverges from price action (price makes new high but basis makes lower high), it signals weakening momentum and potential reversal. Combined with 12h trend filter to avoid counter-trend trades. Works in both bull and bear markets by capturing exhaustion moves. Targets low trade frequency (~15-30/year) to minimize fee drag.
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
    
    # Calculate rolling premium index approximation (close - 20-period EMA)
    # This approximates basis deviation from fair value
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    premium_index = close - ema20  # Positive = premium, Negative = discount
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # 12h trend: bullish when price > EMA34, bearish when price < EMA34
    trend_bull = close > ema34_12h_aligned
    trend_bear = close < ema34_12h_aligned
    
    # Divergence detection: price makes new high/low but premium index doesn't confirm
    # Bearish divergence: price makes higher high, premium index makes lower high
    # Bullish divergence: price makes lower low, premium index makes higher low
    
    # Calculate rolling max/min for divergence detection (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    premium_max = pd.Series(premium_index).rolling(window=20, min_periods=20).max().values
    premium_min = pd.Series(premium_index).rolling(window=20, min_periods=20).min().values
    
    # Bearish divergence: price at recent high but premium index below recent high
    bear_div = (high == high_max) & (premium_index < premium_max)
    # Bullish divergence: price at recent low but premium index above recent low
    bull_div = (low == low_min) & (premium_index > premium_min)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(premium_index[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(premium_max[i]) or np.isnan(premium_min[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: divergence + trend alignment
        bearish_entry = bear_div[i] and trend_bear[i]  # Price high but weak momentum in downtrend
        bullish_entry = bull_div[i] and trend_bull[i]  # Price low but strong momentum in uptrend
        
        # Exit when divergence disappears or trend changes
        bearish_exit = ~bear_div[i] or ~trend_bear[i]
        bullish_exit = ~bull_div[i] or ~trend_bull[i]
        
        if bullish_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif bearish_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif bearish_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif bullish_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Premium_Index_Divergence_12hTrend"
timeframe = "6h"
leverage = 1.0