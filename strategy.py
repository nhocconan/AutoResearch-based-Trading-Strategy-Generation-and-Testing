#!/usr/bin/env python3
# 1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Buy breakouts above Camarilla R1 in uptrends, sell breakdowns below S1 in downtrends on daily timeframe.
# Uses weekly trend filter to avoid counter-trend trades. Targets 10-25 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts) and bear markets (sell breakdowns) by following weekly trend.

name = "1d_1W_Camarilla_R1_S1_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Daily data for Camarilla levels (using previous day's OHLC)
    # Calculate Camarilla levels for today based on yesterday's OHLC
    # We need to shift the OHLC data by 1 to get previous day's values
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first day's previous values to NaN (no previous day)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Calculate Camarilla levels
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    rng = prev_high - prev_low
    r1 = prev_close + 1.1 * rng / 12
    s1 = prev_close - 1.1 * rng / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 in weekly uptrend
            if close[i] > r1[i] and trend_1w_up_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in weekly downtrend
            elif close[i] < s1[i] and trend_1w_down_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below R1 or weekly trend turns down
            if close[i] < r1[i] or trend_1w_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above S1 or weekly trend turns up
            if close[i] > s1[i] or trend_1w_down_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals