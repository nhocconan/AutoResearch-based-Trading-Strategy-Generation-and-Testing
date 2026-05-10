#!/usr/bin/env python3
"""
1D_Wick_Reversal_With_WeeklyTrend
Hypothesis: Price reverses from long upper/lower wicks when weekly trend is aligned.
- Long when: close > open (bullish candle) AND low < weekly EMA50 (wick below trend) AND weekly trend up
- Short when: close < open (bearish candle) AND high > weekly EMA50 (wick above trend) AND weekly trend down
- Exit when: price crosses weekly EMA50 in opposite direction
Works in bull/bear by following weekly trend. Low turnover due to weekly trend filter.
"""

name = "1D_Wick_Reversal_With_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Weekly EMA50 for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_bullish = close[i] > open_price[i]
        is_bearish = close[i] < open_price[i]
        wick_low_below = low[i] < ema50_1w_aligned[i]
        wick_high_above = high[i] > ema50_1w_aligned[i]
        
        trend_up = trend_1w_up_aligned[i] > 0.5
        trend_down = trend_1w_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish candle with wick below weekly EMA50 + weekly uptrend
            if is_bullish and wick_low_below and trend_up:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish candle with wick above weekly EMA50 + weekly downtrend
            elif is_bearish and wick_high_above and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below weekly EMA50
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above weekly EMA50
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals