#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d EMA34 trend + 12h EMA21 entry
# Uses daily EMA34 to determine trend direction, enters on EMA21 cross with choppiness filter
# Choppiness Index (CHOP) > 61.8 indicates ranging (mean revert), < 38.2 indicates trending
# In trending regime (CHOP < 38.2), enter long when price > EMA21, short when price < EMA21
# Designed for 12-37 trades/year with proper risk control via trend failure
name = "12h_EMA21_ChoppinessRegime_1dEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12h EMA21 for entry signal
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 12h Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # First bar has no previous close
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl[range_hl == 0] = 1e-10
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    
    # Align 1d EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 14)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_aligned[i]) or np.isnan(ema21[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade in direction of 1d EMA34
        # Long only when EMA34 is rising, short only when EMA34 is falling
        # We'll use EMA34 slope approximated by current vs previous value
        if i > 0:
            ema34_rising = ema34_aligned[i] > ema34_aligned[i-1]
            ema34_falling = ema34_aligned[i] < ema34_aligned[i-1]
        else:
            ema34_rising = True
            ema34_falling = False
        
        # Chop regime filter: trending when CHOP < 38.2
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Look for entry: price crosses EMA21 in trending regime with trend filter
            # Long: price crosses above EMA21
            if close[i] > ema21[i] and close[i-1] <= ema21[i-1] and is_trending and ema34_rising:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below EMA21
            elif close[i] < ema21[i] and close[i-1] >= ema21[i-1] and is_trending and ema34_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA21 or trend changes
            if close[i] < ema21[i] and close[i-1] >= ema21[i-1]:
                signals[i] = 0.0
                position = 0
            elif not (is_trending and ema34_rising):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA21 or trend changes
            if close[i] > ema21[i] and close[i-1] <= ema21[i-1]:
                signals[i] = 0.0
                position = 0
            elif not (is_trending and ema34_falling):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals