#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly trend filter (EMA200) and daily Bollinger Band squeeze breakout.
# Uses weekly EMA200 for trend direction, daily Bollinger Bands for volatility squeeze detection,
# and 12h price action for breakout confirmation. Designed to work in both bull and bear markets
# by capturing volatility expansions after low-volatility periods, with trend filter preventing
# counter-trend trades. Targets 15-30 trades/year to avoid fee drag.

name = "12h_WeeklyEMA200_Trend_DailyBBSqueeze_Breakout"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly EMA200 for trend
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily Bollinger Bands (20, 2.0)
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + 2 * std20_1d
    lower_bb = sma20_1d - 2 * std20_1d
    bb_width = (upper_bb - lower_bb) / sma20_1d  # Normalized width
    
    # Align weekly EMA200 to 12h
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Align daily Bollinger Bands and width to 12h
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Bollinger Band squeeze detection: width below 20-period percentile
    bb_width_ma = pd.Series(bb_width_aligned).rolling(window=20, min_periods=20).mean().values
    squeeze_condition = bb_width_aligned < (bb_width_ma * 0.8)  # 20% below average width
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(squeeze_condition[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: bullish weekly trend, BB squeeze breakout above upper band
            if (close[i] > ema200_1w_aligned[i] and 
                close[i] > upper_bb_aligned[i] and 
                squeeze_condition[i]):
                signals[i] = 0.25
                position = 1
            # Short: bearish weekly trend, BB squeeze breakout below lower band
            elif (close[i] < ema200_1w_aligned[i] and 
                  close[i] < lower_bb_aligned[i] and 
                  squeeze_condition[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly EMA200 or below lower BB
            if close[i] < ema200_1w_aligned[i] or close[i] < lower_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly EMA200 or above upper BB
            if close[i] > ema200_1w_aligned[i] or close[i] > upper_bb_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals