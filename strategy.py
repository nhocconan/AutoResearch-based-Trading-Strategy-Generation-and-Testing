#!/usr/bin/env python3
"""
6h_MultiTimeframe_Bollinger_Bands_With_Trend_Filter
Hypothesis: Combine 1d Bollinger Band squeeze (low volatility) with 1w trend direction on 6h timeframe to capture breakouts from consolidation. Works in both bull and bear markets by using volatility contraction as entry signal and higher timeframe trend for direction.
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
    volume = prices['volume'].values
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(df_1d['close']).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (std_20 * bb_std)
    lower_bb = sma_20 - (std_20 * bb_std)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 6h
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Band squeeze condition (low volatility)
    bb_width_ma = pd.Series(bb_width_aligned).rolling(window=10, min_periods=10).mean().values
    squeeze_condition = bb_width_aligned < bb_width_ma  # Width below its MA
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    w1_uptrend = close > ema_50_1w_aligned
    w1_downtrend = close < ema_50_1w_aligned
    
    # Breakout conditions
    long_breakout = close > upper_bb_aligned
    short_breakout = close < lower_bb_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: breakout from Bollinger Band squeeze with trend alignment
        long_entry = long_breakout[i] and squeeze_condition[i] and w1_uptrend[i]
        short_entry = short_breakout[i] and squeeze_condition[i] and w1_downtrend[i]
        
        # Exit conditions: opposite breakout or loss of squeeze
        long_exit = short_breakout[i] or not squeeze_condition[i]
        short_exit = long_breakout[i] or not squeeze_condition[i]
        
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

name = "6h_MultiTimeframe_Bollinger_Bands_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0