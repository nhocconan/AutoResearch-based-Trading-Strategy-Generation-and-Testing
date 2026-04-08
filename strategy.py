#!/usr/bin/env python3
# 4h_rsi_pullback_12h_trend_v1
# Hypothesis: On 4h timeframe, buy pullbacks in uptrend and sell rallies in downtrend using RSI(14) with 12h trend filter.
# Long when RSI < 30 and price > 12h EMA(50). Short when RSI > 70 and price < 12h EMA(50).
# Uses 12h EMA for trend filter to avoid counter-trend trades. Designed for 20-50 trades/year.
# Works in bull markets via buying dips and bear markets via selling rallies.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_rsi_pullback_12h_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA(50) is ready
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or trend reverses (price < EMA)
            if rsi[i] > 70 or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 or trend reverses (price > EMA)
            if rsi[i] < 30 or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI < 30 and price > EMA (pullback in uptrend)
            if rsi[i] < 30 and close[i] > ema50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 and price < EMA (rally in downtrend)
            elif rsi[i] > 70 and close[i] < ema50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals