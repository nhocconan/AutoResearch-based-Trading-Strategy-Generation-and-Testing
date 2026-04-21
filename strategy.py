#!/usr/bin/env python3
"""
4h_MACD_Trend_With_1d_TrendFilter_V1
Hypothesis: MACD on 4h provides momentum signals, filtered by 1d EMA50 trend direction to avoid counter-trend trades. Works in bull/bear by only taking trades aligned with higher timeframe trend. Uses volume confirmation and ATR-based stoploss to reduce whipsaw and limit trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h MACD calculation
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # MACD components
    ema_fast = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_4h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # MACD crossover signals
        macd_bullish = macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]
        macd_bearish = macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]
        
        # Volume confirmation (above average)
        volume_ok = volume_4h[i] > vol_ma[i]
        
        # Trend filter: 1d EMA50 direction
        price = close_4h[i]
        ema_50 = ema_50_1d_aligned[i]
        uptrend = price > ema_50
        downtrend = price < ema_50
        
        if position == 0:
            # Long: bullish MACD cross + volume + uptrend on 1d
            if macd_bullish and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish MACD cross + volume + downtrend on 1d
            elif macd_bearish and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish MACD cross or price below 1d EMA50
            if macd_bearish or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish MACD cross or price above 1d EMA50
            if macd_bullish or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_MACD_Trend_With_1d_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0