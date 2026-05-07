#!/usr/bin/env python3
"""
1d_WeeklyCandlePattern_1WTrend_Filter
Hypothesis: Use weekly candle patterns (engulfing/bullish/bearish) on the close of weekly candles,
filtered by the trend of the weekly EMA20, to capture multi-week trends in BTC/ETH.
This strategy aims for low trade frequency (<10/year per symbol) with high conviction entries,
making it suitable for the 1d timeframe and avoiding fee drag. Works in bull and bear by
trading with the weekly trend: long when weekly close > weekly EMA20 and bullish engulfing,
short when weekly close < weekly EMA20 and bearish engulfing.
"""
name = "1d_WeeklyCandlePattern_1WTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    
    # Get weekly data for trend filter and pattern detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly bullish and bearish engulfing patterns
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Bullish engulfing: current week bullish (close > open) and engulfs previous week's body
    bullish_engulf = (close_1w > open_1w) & (close_1w >= open_1w[:-1]) & (open_1w <= close_1w[:-1])
    # Bearish engulfing: current week bearish (close < open) and engulfs previous week's body
    bearish_engulf = (close_1w < open_1w) & (close_1w <= open_1w[:-1]) & (open_1w >= close_1w[:-1])
    
    # Shift to align with current week's signal (pattern known at weekly close)
    bullish_engulf = np.roll(bullish_engulf, 1)
    bearish_engulf = np.roll(bearish_engulf, 1)
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # Align patterns to daily timeframe
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bearish_engulf.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA20 warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(bullish_engulf_aligned[i]) or 
            np.isnan(bearish_engulf_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: weekly close above weekly EMA20 and bullish engulfing pattern
            if close[i] > ema_20_1w_aligned[i] and bullish_engulf_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: weekly close below weekly EMA20 and bearish engulfing pattern
            elif close[i] < ema_20_1w_aligned[i] and bearish_engulf_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly close crosses below weekly EMA20
            if close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly close crosses above weekly EMA20
            if close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals