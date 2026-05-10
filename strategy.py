#!/usr/bin/env python3
# 1D_1W_Donchian_Breakout_Trend_Follow
# Hypothesis: In strong weekly trends, price breaks daily Donchian channels to capture trends while avoiding whipsaws.
# Long when price breaks above daily Donchian high (20) in a weekly uptrend (close > weekly EMA50).
# Short when price breaks below daily Donchian low (20) in a weekly downtrend (close < weekly EMA50).
# Weekly trend filter reduces false breaks in ranging markets. Works in bull/bear by following weekly trend.
# Target: 15-25 trades/year per symbol.

name = "1D_1W_Donchian_Breakout_Trend_Follow"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1w > ema50_1w
    bearish_trend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    # Daily Donchian channel (20-period)
    lookback = 20
    # Calculate highest high and lowest low over lookback period
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(50, lookback - 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish weekly trend + price breaks above daily Donchian high
            if bullish and close[i] > highest_high[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish weekly trend + price breaks below daily Donchian low
            elif bearish and close[i] < lowest_low[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish weekly trend or price breaks below daily Donchian low
            if bearish or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish weekly trend or price breaks above daily Donchian high
            if bullish or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals