#!/usr/bin/env python3
# 1d_KAMA_With_1wTrend
# Hypothesis: In multi-year crypto markets, the 1-week trend provides a strong filter for daily trades.
# We use 1-week EMA50 as the primary trend filter and enter on daily KAMA direction aligned with the weekly trend.
# Long when: 1w trend up (close > EMA50_1w) AND KAMA(14,2,30) is rising.
# Short when: 1w trend down (close < EMA50_1w) AND KAMA(14,2,30) is falling.
# This captures continuation moves in trending markets while avoiding counter-trend trades.
# Works in both bull (follows strong uptrends) and bear (follows strong downtrends).
# Uses volume confirmation to avoid low-conviction breakouts.

name = "1d_KAMA_With_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on daily chart
    close_s = pd.Series(close)
    # Efficiency ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(10).sum()
    er = change / volatility.replace(0, np.nan)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation (20-day MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need KAMA (30), EMA50_1w (50), volume MA (20)
    start_idx = max(30, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # KAMA direction
        if i > 0:
            kama_rising = kama[i] > kama[i-1]
            kama_falling = kama[i] < kama[i-1]
        else:
            kama_rising = False
            kama_falling = False
        
        if position == 0:
            # Long entry: uptrend + KAMA rising + volume
            if uptrend and kama_rising and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + KAMA falling + volume
            elif downtrend and kama_falling and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or KAMA turns down
            if not uptrend or not kama_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or KAMA turns up
            if not downtrend or not kama_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals