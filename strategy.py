#!/usr/bin/env python3
"""
6h_1d_wk_rsi_divergence_v1
Strategy: 6s RSI divergence with 1d/1w trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses RSI(14) divergence on 6h chart (bullish: price makes lower low, RSI makes higher low; bearish: price makes higher high, RSI makes lower high) confirmed by 1d EMA50 and 1w EMA200 trend alignment. Designed to catch reversals in both bull and bear markets by combining momentum divergence with higher timeframe trend filters. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_wk_rsi_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 6h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Detect RSI divergence (lookback 5 periods)
    def detect_divergence(high_arr, low_arr, rsi_arr, lookback=5):
        bull_div = np.zeros(len(close), dtype=bool)
        bear_div = np.zeros(len(close), dtype=bool)
        
        for i in range(lookback, len(close)):
            # Bullish divergence: price makes lower low, RSI makes higher low
            if low_arr[i] < low_arr[i-lookback:i].min() and rsi_arr[i] > rsi_arr[i-lookback:i].min():
                bull_div[i] = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            if high_arr[i] > high_arr[i-lookback:i].max() and rsi_arr[i] < rsi_arr[i-lookback:i].max():
                bear_div[i] = True
        return bull_div, bear_div
    
    bull_div, bear_div = detect_divergence(high, low, rsi, 5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filters: price above both EMAs for long, below both for short
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        uptrend_1w = price_close > ema_200_1w_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        downtrend_1w = price_close < ema_200_1w_aligned[i]
        
        # Divergence signals
        bull_signal = bull_div[i] and uptrend_1d and uptrend_1w
        bear_signal = bear_div[i] and downtrend_1d and downtrend_1w
        
        # Exit when divergence fails or opposite signal appears
        exit_long = position == 1 and (not bull_div[i] or bear_div[i])
        exit_short = position == -1 and (not bear_div[i] or bull_div[i])
        
        # Trading logic
        if bull_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bear_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals