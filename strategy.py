#!/usr/bin/env python3
# 1D_1W_Camarilla_R1_S1_Breakout_Trend_Filter
# Hypothesis: Price reacts strongly to weekly Camarilla pivot levels (R1/S1) derived from weekly range.
# Long when price closes above weekly R1 in a weekly uptrend (close > EMA50 weekly).
# Short when price closes below weekly S1 in a weekly downtrend (close < EMA50 weekly).
# Uses weekly EMA50 for trend filter and weekly Camarilla R1/S1 for entry.
# Works in bull/bear by following weekly trend direction. Target: 10-20 trades/year per symbol.

name = "1D_1W_Camarilla_R1_S1_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1w - low_1w) * 1.1 / 12
    r1_1w = close_1w + camarilla_range
    s1_1w = close_1w - camarilla_range
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1w > ema50_1w
    bearish_trend = close_1w < ema50_1w
    
    # Align to daily
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish trend + price closes above weekly R1
            if bullish and close[i] > r1_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish trend + price closes below weekly S1
            elif bearish and close[i] < s1_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price closes below weekly EMA50
            ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
            if bearish or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price closes above weekly EMA50
            ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
            if bullish or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals