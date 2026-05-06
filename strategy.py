#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly RSI mean reversion with daily trend filter
# Weekly RSI > 70 indicates overbought conditions (short opportunity), < 30 indicates oversold (long opportunity)
# Daily EMA 50 acts as trend filter: only take longs when price > daily EMA 50, shorts when price < daily EMA 50
# This combines mean reversion on higher timeframe with trend alignment on lower timeframe
# Works in both bull/bear markets: mean reversion captures pullbacks in trends, trend filter avoids counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyRSI30_70_DailyEMA50_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate weekly RSI ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly RSI calculation
    delta = pd.Series(df_1w['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align weekly RSI to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi_values)
    
    # Calculate daily EMA 50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long signal: weekly RSI < 30 (oversold) and price > daily EMA 50 (uptrend)
            if rsi_aligned[i] < 30 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short signal: weekly RSI > 70 (overbought) and price < daily EMA 50 (downtrend)
            elif rsi_aligned[i] > 70 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: weekly RSI > 50 (mean reversion complete) or price < daily EMA 50 (trend change)
            if rsi_aligned[i] > 50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: weekly RSI < 50 (mean reversion complete) or price > daily EMA 50 (trend change)
            if rsi_aligned[i] < 50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals