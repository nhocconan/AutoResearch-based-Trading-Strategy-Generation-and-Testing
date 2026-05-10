#!/usr/bin/env python3
# 1D_1W_Camarilla_R3_S3_Breakout_Trend
# Hypothesis: Camarilla pivot levels on daily chart act as strong support/resistance.
# Breakout above R3 or below S3 with weekly trend continuation signals strong momentum.
# Uses weekly EMA34 for trend filter to avoid counter-trend trades.
# Works in bull/bear by following weekly trend. Target: 10-20 trades/year per symbol.

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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels for each day (based on previous day's OHLC)
    R3 = np.zeros(len(close_1d))
    S3 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        prev_close = close_1d[i-1]
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R3[i] = prev_close + (range_ * 1.1 / 4)
            S3[i] = prev_close - (range_ * 1.1 / 4)
        else:
            R3[i] = prev_close
            S3[i] = prev_close
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Trend: bullish if close > EMA34, bearish if close < EMA34
    bullish_trend = close_1w > ema34_1w
    bearish_trend = close_1w < ema34_1w
    
    # Align daily Camarilla levels to 1d timeframe (already aligned as daily data)
    # Align weekly trend to 1d timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 1  # Need previous day for Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: bullish weekly trend + price breaks above R3
            if bullish and high[i] > R3[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish weekly trend + price breaks below S3
            elif bearish and low[i] < S3[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish weekly trend or price drops below S3
            if bearish or low[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish weekly trend or price rises above R3
            if bullish or high[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals