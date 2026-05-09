#!/usr/bin/env python3

# Hypothesis: 6h timeframe with daily volatility breakout and weekly trend filter.
# Uses daily ATR-based volatility breakout (price > close + k*ATR for long, price < close - k*ATR for short)
# combined with weekly EMA50 trend filter to avoid counter-trend trades.
# Volatility breakouts capture momentum bursts, while weekly trend filter reduces whipsaw in ranging markets.
# Designed to work in both bull and bear markets by filtering trades with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "6h_ATRBreakout_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR(14) for volatility breakout
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility breakout: k=0.5 (breakout when price moves 0.5x ATR from close)
    k = 0.5
    breakout_up = close > (close + k * atr)
    breakout_down = close < (close - k * atr)
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(breakout_up[i]) or np.isnan(breakout_down[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: volatility breakout up + weekly uptrend
            if breakout_up[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: volatility breakout down + weekly downtrend
            elif breakout_down[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: volatility contraction or trend reversal
            if close[i] < (close[i-1] + 0.25 * atr[i]) or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: volatility contraction or trend reversal
            if close[i] > (close[i-1] - 0.25 * atr[i]) or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals