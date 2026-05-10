#!/usr/bin/env python3
# 1D_1W_Camarilla_R3_S3_Breakout_Trend
# Hypothesis: On daily timeframe, price breaks Camarilla R3/S3 levels with weekly trend filter and volume confirmation.
# Long when price closes above R3 in weekly uptrend (weekly close > weekly EMA34).
# Short when price closes below S3 in weekly downtrend (weekly close < weekly EMA34).
# Uses daily Camarilla levels for entry and weekly EMA34 for trend filter.
# Works in bull/bear by following weekly trend direction. Target: 15-25 trades/year per symbol.

name = "1D_1W_Camarilla_R3_S3_Breakout_Trend"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    diff_1d = high_1d - low_1d
    R3_1d = close_1d + 1.1 * diff_1d / 2
    S3_1d = close_1d - 1.1 * diff_1d / 2
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Trend: bullish if weekly close > EMA34, bearish if weekly close < EMA34
    bullish_trend_1w = close_1w > ema34_1w
    bearish_trend_1w = close_1w < ema34_1w
    
    # Align all to daily timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend_1w.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend_1w.astype(float))
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish weekly trend + price closes above R3 + volume confirmation
            if bullish and close[i] > R3_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish weekly trend + price closes below S3 + volume confirmation
            elif bearish and close[i] < S3_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish trend or price closes below R3
            if bearish or close[i] < R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish trend or price closes above S3
            if bullish or close[i] > S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals