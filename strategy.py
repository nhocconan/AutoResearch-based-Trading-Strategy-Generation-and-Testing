#!/usr/bin/env python3
# 1D_1W_CCI_Trend_Follow
# Hypothesis: On the daily chart, CCI(20) identifies overbought/oversold conditions within the weekly trend.
# In a weekly uptrend (close > weekly EMA50), go long when daily CCI crosses below -100 (oversold pullback).
# In a weekly downtrend (close < weekly EMA50), go short when daily CCI crosses above 100 (overbought pullback).
# Uses weekly EMA50 for trend filter and daily CCI(20) for entry timing.
# Works in bull/bear by following weekly trend direction. Target: 10-20 trades/year per symbol.

name = "1D_1W_CCI_Trend_Follow"
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
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Trend: bullish if close > EMA50, bearish if close < EMA50
    bullish_trend = close_1w > ema50_1w
    bearish_trend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend.astype(float))
    
    # Daily CCI(20)
    typical_price = (high + low + close) / 3.0
    tp_series = pd.Series(typical_price)
    sma_tp = tp_series.rolling(window=20, min_periods=20).mean()
    mad = tp_series.rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (typical_price - sma_tp.values) / (0.015 * mad.values)
    # Handle division by zero or invalid mad
    cci = np.where(mad.values == 0, 0, cci)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i]) or
            np.isnan(cci[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bullish = bullish_aligned[i] > 0.5
        bearish = bearish_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + daily CCI crosses below -100 (oversold)
            if bullish and cci[i] < -100 and cci[i-1] >= -100:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + daily CCI crosses above 100 (overbought)
            elif bearish and cci[i] > 100 and cci[i-1] <= 100:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly downtrend or daily CCI crosses above 100 (overbought)
            if bearish or (cci[i] > 100 and cci[i-1] <= 100):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly uptrend or daily CCI crosses below -100 (oversold)
            if bullish or (cci[i] < -100 and cci[i-1] >= -100):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals