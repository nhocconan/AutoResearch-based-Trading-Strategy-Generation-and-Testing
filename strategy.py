#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session filter.
In strong 4h trends (above/below 200 EMA), extreme 1h RSI readings often reverse.
Uses 4h EMA200 for trend direction, 1h RSI(2) for entry timing, and 08-20 UTC session filter.
Designed for low trade frequency (target: 15-30/year) to avoid fee drag.
Works in both bull and bear markets by following 4h trend while fading short-term extremes.
"""

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
    
    # 4h EMA200 for trend direction
    df_4h = get_htf_data(prices, '4h')
    ema200_4h = pd.Series(df_4h['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # 1h RSI(2) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (same as RSI)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        price = close[i]
        rsi_val = rsi_values[i]
        ema200 = ema200_4h_aligned[i]
        
        if np.isnan(rsi_val) or np.isnan(ema200):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > EMA200) + oversold RSI(2) < 10
            if price > ema200 and rsi_val < 10:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < EMA200) + overbought RSI(2) > 90
            elif price < ema200 and rsi_val > 90:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (> 50) or trend reversal
            if rsi_val > 50 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI returns to neutral (< 50) or trend reversal
            if rsi_val < 50 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_TrendFilter_Session"
timeframe = "1h"
leverage = 1.0