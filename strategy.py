#!/usr/bin/env python3
"""
1d_1W_Trend_Following_Strategy
Hypothesis: Use weekly trend (1w EMA50) as filter and daily price action for entries.
In bull markets: buy pullbacks to EMA20 on dips in uptrend.
In bear markets: sell rallies to EMA20 in downtrend.
Add volume confirmation to avoid false signals. Target: 10-25 trades/year.
Works in both bull and bear by following the higher timeframe trend.
"""

name = "1d_1W_Trend_Following_Strategy"
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
    
    # Daily EMA20 for entry signals
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly trend filter: EMA50 on 1w data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1w = df_1w['close'].values > ema_50_1w
    downtrend_1w = df_1w['close'].values < ema_50_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        ema20 = ema_20[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: price near EMA20 in uptrend with volume
            if uptrend and vol_conf and close[i] <= ema20 * 1.01 and close[i] >= ema20 * 0.99:
                signals[i] = 0.25
                position = 1
            # SHORT: price near EMA20 in downtrend with volume
            elif downtrend and vol_conf and close[i] <= ema20 * 1.01 and close[i] >= ema20 * 0.99:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: trend reversal or price moves away from EMA20
            if not uptrend or close[i] > ema20 * 1.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: trend reversal or price moves away from EMA20
            if not downtrend or close[i] < ema20 * 0.95:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals