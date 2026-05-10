#!/usr/bin/env python3
# 1h_4H1D_Trend_Retracement
# Hypothesis: In 4h and 1d trends, price retraces to EMA21 on 1h before continuing.
# Enter long in 4h/1d uptrend when 1h price touches EMA21 with rejection candle.
# Enter short in 4h/1d downtrend when 1h price touches EMA21 with rejection candle.
# Uses 4h/1d for trend direction, 1h for entry timing. Reduces frequency vs pure 1h trend.

name = "1h_4H1D_Trend_Retracement"
timeframe = "1h"
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
    open_price = prices['open'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on 4h close
    ema21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate EMA21 on 1d close
    ema21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Calculate EMA21 on 1h for entry
    ema21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA21 (21), 1d EMA21 (21), 1h EMA21 (21)
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(ema21_1d_aligned[i]) or 
            np.isnan(ema21_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h and 1d trend
        trend_4h_up = close[i] > ema21_4h_aligned[i]
        trend_4h_down = close[i] < ema21_4h_aligned[i]
        trend_1d_up = close[i] > ema21_1d_aligned[i]
        trend_1d_down = close[i] < ema21_1d_aligned[i]
        
        # Both 4h and 1d must agree on trend
        uptrend = trend_4h_up and trend_1d_up
        downtrend = trend_4h_down and trend_1d_down
        
        # 1h rejection candle: close near high for bullish rejection, near low for bearish
        body_size = abs(close[i] - open_price[i])
        range_size = high[i] - low[i]
        if range_size > 0:
            upper_wick = high[i] - max(close[i], open_price[i])
            lower_wick = min(close[i], open_price[i]) - low[i]
            # Bullish rejection: long lower wick, small body, close near high
            bullish_rejection = (lower_wick > body_size * 1.5) and (close[i] > open_price[i]) and (upper_wick < body_size * 0.5)
            # Bearish rejection: long upper wick, small body, close near low
            bearish_rejection = (upper_wick > body_size * 1.5) and (close[i] < open_price[i]) and (lower_wick < body_size * 0.5)
        else:
            bullish_rejection = False
            bearish_rejection = False
        
        if position == 0:
            # Long entry: uptrend on 4h/1d + price at EMA21(1h) + bullish rejection
            if uptrend and abs(close[i] - ema21_1h[i]) / ema21_1h[i] < 0.005 and bullish_rejection:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend on 4h/1d + price at EMA21(1h) + bearish rejection
            elif downtrend and abs(close[i] - ema21_1h[i]) / ema21_1h[i] < 0.005 and bearish_rejection:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price moves far from EMA21
            if not uptrend or abs(close[i] - ema21_1h[i]) / ema21_1h[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price moves far from EMA21
            if not downtrend or abs(close[i] - ema21_1h[i]) / ema21_1h[i] > 0.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals