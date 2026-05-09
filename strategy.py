#!/usr/bin/env python3
# 1h_TrendPullback_4hTrend_RSI_Entry
# Hypothesis: Trend-following pullback strategy. Use 4h EMA trend for direction, enter on 1h pullbacks (RSI < 40 long / > 60 short). 
# Works in bull/bear: trend filter avoids counter-trend trades, RSI provides precise entry timing.
# Focus on low-frequency, high-probability entries to minimize fee drag.

name = "1h_TrendPullback_4hTrend_RSI_Entry"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA21 for trend
    ema_21_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 21:
        ema_21_4h[20] = np.mean(close_4h[0:21])
        for i in range(21, len(close_4h)):
            ema_21_4h[i] = (ema_21_4h[i-1] * 20 + close_4h[i]) / 21
    
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure RSI is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(rsi[i]) or not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend (price > EMA21) and RSI < 40 (oversold pullback)
            if close[i] > ema_21_4h_aligned[i] and rsi[i] < 40:
                signals[i] = 0.20
                position = 1
            # Enter short: downtrend (price < EMA21) and RSI > 60 (overbought pullback)
            elif close[i] < ema_21_4h_aligned[i] and rsi[i] > 60:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or RSI > 70 (overbought)
            if close[i] < ema_21_4h_aligned[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend reversal or RSI < 30 (oversold)
            if close[i] > ema_21_4h_aligned[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals