#!/usr/bin/env python3
"""
6h_StructureBreak_WeeklyPivot_Trend
Hypothesis: Combines weekly pivot points with price structure breaks (HH/HL/LH/LL) and multi-timeframe trend alignment. 
Uses weekly pivot levels from 1w data as support/resistance zones. 
Goes long when price makes a higher low above weekly pivot support with bullish alignment across 6h/1d/1w timeframes.
Goes short when price makes a lower high below weekly pivot resistance with bearish alignment.
Designed for low-frequency, high-conviction trades targeting sustained moves in both bull and bear markets.
Target: 20-40 trades/year per symbol.
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
    
    # Get weekly data for pivot points and trend
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 10 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # Support 1: S1 = (2*P) - H
    # Resistance 1: R1 = (2*P) - L
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = (2 * pivot) - prev_week_high
    s1 = (2 * pivot) - prev_week_low
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Trend filters: 6x EMA for short-term, daily EMA50 for intermediate, weekly EMA20 for long-term bias
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Trend alignment conditions
    # Short-term trend: price > EMA6
    st_uptrend = close > ema_6
    st_downtrend = close < ema_6
    
    # Intermediate trend: price > daily EMA50
    int_uptrend = close > ema_50_1d_aligned
    int_downtrend = close < ema_50_1d_aligned
    
    # Long-term bias: price > weekly EMA20
    lt_bullish = close > ema_20_1w_aligned
    lt_bearish = close < ema_20_1w_aligned
    
    # Price structure: detect higher lows and lower highs
    # Higher low: current low > prior low AND current close > prior close
    hl_condition = (low > np.roll(low, 1)) & (close > np.roll(close, 1))
    # Lower high: current high < prior high AND current close < prior close
    lh_condition = (high < np.roll(high, 1)) & (close < np.roll(close, 1))
    
    # Align structure signals (they're already 6h, no need for HTF alignment)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        # Long: Higher low structure above weekly support + bullish alignment across timeframes
        long_entry = (hl_condition[i] and 
                     low[i] > s1_aligned[i] and 
                     st_uptrend[i] and 
                     int_uptrend[i] and 
                     lt_bullish[i])
        
        # Short: Lower high structure below weekly resistance + bearish alignment across timeframes
        short_entry = (lh_condition[i] and 
                      high[i] < r1_aligned[i] and 
                      st_downtrend[i] and 
                      int_downtrend[i] and 
                      lt_bearish[i])
        
        # Exit conditions: trend breakdown or structure failure
        long_exit = (not st_uptrend[i]) or (not int_uptrend[i]) or (close[i] < s1_aligned[i])
        short_exit = (not st_downtrend[i]) or (not int_downtrend[i]) or (close[i] > r1_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
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

name = "6h_StructureBreak_WeeklyPivot_Trend"
timeframe = "6h"
leverage = 1.0