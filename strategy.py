#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Williams %R with trend filter from 6h EMA
# Long when daily Williams %R < -80 (oversold) and price > 6h EMA50 (bullish trend)
# Short when daily Williams %R > -20 (overbought) and price < 6h EMA50 (bearish trend)
# Williams %R identifies overextended moves, EMA50 filters for trend direction
# Works in both bull/bear markets by fading extremes in the direction of trend
# Target: 50-100 total trades over 4 years (~12-25/year) with 0.25 position sizing

name = "6h_1dWilliamsR_EMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 6h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50[i]) or np.isnan(williams_r_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: oversold Williams %R + bullish trend (price > EMA50)
            if williams_r_aligned[i] < -80 and close[i] > ema50[i]:
                signals[i] = 0.25
                position = 1
            # Short: overbought Williams %R + bearish trend (price < EMA50)
            elif williams_r_aligned[i] > -20 and close[i] < ema50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R exits oversold OR trend turns bearish
            if williams_r_aligned[i] > -50 or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R exits overbought OR trend turns bullish
            if williams_r_aligned[i] < -50 or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals