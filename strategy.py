#!/usr/bin/env python3
# 6h_ChandelierExit_Breakout_12hTrend_Volume
# Hypothesis: Long when price breaks above Chandelier Exit long, above 12h EMA50, with volume spike; short when breaks below Chandelier Exit short, below 12h EMA50, with volume spike.
# Chandelier Exit adapts to volatility, providing dynamic support/resistance. EMA50 filters trend direction, volume confirms breakout strength.
# Designed for 6h to achieve 12-37 trades/year, suitable for both bull and bear markets by adapting to volatility and using trend filter.

name = "6h_ChandelierExit_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chandelier Exit parameters
    atr_period = 22
    multiplier = 3.0
    
    # True Range and ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(atr_period, n):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Chandelier Exit Long and Short
    chandelier_long = np.full(n, np.nan)
    chandelier_short = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    highest_high[0] = high[0]
    lowest_low[0] = low[0]
    for i in range(1, n):
        highest_high[i] = max(highest_high[i-1], high[i])
        lowest_low[i] = min(lowest_low[i-1], low[i])
    
    for i in range(atr_period, n):
        if not np.isnan(atr[i]):
            chandelier_long[i] = highest_high[i] - multiplier * atr[i]
            chandelier_short[i] = lowest_low[i] + multiplier * atr[i]
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(chandelier_long[i]) or np.isnan(chandelier_short[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Chandelier Long, above 12h EMA50, strong volume
            if close[i] > chandelier_long[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Chandelier Short, below 12h EMA50, strong volume
            elif close[i] < chandelier_short[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 2.0 * vol_ma_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Chandelier Long or below 12h EMA50
            if close[i] < chandelier_long[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Chandelier Short or above 12h EMA50
            if close[i] > chandelier_short[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals