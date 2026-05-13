#!/usr/bin/env python3
"""
6h_RSI_Extreme_Trend_Filter
Hypothesis: Use RSI(14) extremes (overbought/oversold) combined with trend filters from higher timeframes. 
Go long when RSI < 30 (oversold) and price > daily EMA50, short when RSI > 70 (overbought) and price < daily EMA50. 
Use weekly trend filter (price > weekly EMA200 for longs, price < weekly EMA200 for shorts) to avoid counter-trend trades.
This captures mean reversion in strong trends, working in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Designed for 6h timeframe to limit trades and avoid fee drag.
"""

name = "6h_RSI_Extreme_Trend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate RSI(14) on 6h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) and price above daily EMA50 and weekly EMA200
            if rsi[i] < 30 and close[i] > ema_50_1d_aligned[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) and price below daily EMA50 and weekly EMA200
            elif rsi[i] > 70 and close[i] < ema_50_1d_aligned[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>45) or price breaks below daily EMA50
            if rsi[i] > 45 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<55) or price breaks above daily EMA50
            if rsi[i] < 55 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals