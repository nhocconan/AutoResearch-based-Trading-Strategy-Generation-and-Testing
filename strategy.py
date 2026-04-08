#!/usr/bin/env python3
# 6h_1w_1d_price_action_breakout_v1
# Hypothesis: Combining weekly price structure (higher highs/lows) with daily momentum
# and 6-hour breakout triggers creates a trend-following strategy that works in both
# bull and bear markets. Weekly structure filters trades to the dominant trend,
# daily RSI filters for momentum exhaustion, and 6-hour Donchian breakouts provide
# precise entry timing. This reduces whipsaws by requiring alignment across three
# timeframes while keeping trade frequency low (target: 15-30 trades/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_price_action_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend structure (higher highs/lows)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly higher highs and higher lows
    whh = df_1w['high'].rolling(window=2, min_periods=2).max().values  # Higher high
    whl = df_1w['low'].rolling(window=2, min_periods=2).min().values   # Higher low
    lhh = df_1w['high'].rolling(window=2, min_periods=2).min().values  # Lower high
    lhl = df_1w['low'].rolling(window=2, min_periods=2).max().values   # Lower low
    
    # Weekly trend: bullish if making higher highs and higher lows
    weekly_bullish = (whh > np.roll(whh, 1)) & (whl > np.roll(whl, 1))
    # Weekly trend: bearish if making lower highs and lower lows
    weekly_bearish = (lhh < np.roll(lhh, 1)) & (lhl < np.roll(lhl, 1))
    
    # Align weekly trends to 6h
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Get daily data for momentum filter (RSI)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 6-hour Donchian breakout (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(weekly_bullish_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly trend turns bearish OR price breaks below Donchian low
            if weekly_bearish_aligned[i] > 0.5 or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: weekly trend turns bullish OR price breaks above Donchian high
            if weekly_bullish_aligned[i] > 0.5 or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: weekly bullish, daily RSI not overbought, break above Donchian high
            if (weekly_bullish_aligned[i] > 0.5 and 
                rsi_1d_aligned[i] < 70 and 
                close[i] > highest_high[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: weekly bearish, daily RSI not oversold, break below Donchian low
            elif (weekly_bearish_aligned[i] > 0.5 and 
                  rsi_1d_aligned[i] > 30 and 
                  close[i] < lowest_low[i]):
                position = -1
                signals[i] = -0.25
    
    return signals