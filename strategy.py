#!/usr/bin/env python3
# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and 1d ATR-based volatility filter.
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND 1d ATR ratio (current/20-period MA) < 1.2 (low volatility environment).
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND 1d ATR ratio < 1.2.
# Exit when price touches Donchian midpoint OR ATR ratio > 1.5 (volatility expansion).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts in low volatility regimes
# and avoiding whipsaws during high volatility periods. The 12h EMA ensures alignment with medium-term trend.

name = "4h_DonchianBreakout_TrendVolFilter_v1"
timeframe = "4h"
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d ATR for volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i],
                   abs(high_1d[i] - close_1d[i-1]),
                   abs(low_1d[i] - close_1d[i-1]))
    tr[0] = high_1d[0] - low_1d[0]
    
    # ATR(20)
    atr_period = 20
    if len(tr) < atr_period:
        return np.zeros(n)
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # ATR ratio: current ATR / 20-period MA of ATR
    atr_ma_period = 20
    if len(atr) < atr_ma_period:
        atr_ratio = np.ones_like(atr)  # default to 1.0 (neutral) if insufficient data
    else:
        atr_ma = np.zeros_like(atr)
        atr_ma[atr_ma_period-1] = np.mean(atr[:atr_ma_period])
        for i in range(atr_ma_period, len(atr)):
            atr_ma[i] = (atr_ma[i-1] * (atr_ma_period-1) + atr[i]) / atr_ma_period
        atr_ratio = atr / atr_ma
        atr_ratio = np.where(atr_ma == 0, 1.0, atr_ratio)  # avoid division by zero
    
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period)
    lookback = 20
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling max/min
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # Donchian midpoint
    midpoint = (highest + lowest) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above Donchian high AND price > 12h EMA50 AND low volatility (ATR ratio < 1.2)
            if (close[i] > highest[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                atr_ratio_aligned[i] < 1.2):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low AND price < 12h EMA50 AND low volatility (ATR ratio < 1.2)
            elif (close[i] < lowest[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  atr_ratio_aligned[i] < 1.2):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches Donchian midpoint OR high volatility (ATR ratio > 1.5)
            if (close[i] <= midpoint[i] or 
                atr_ratio_aligned[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches Donchian midpoint OR high volatility (ATR ratio > 1.5)
            if (close[i] >= midpoint[i] or 
                atr_ratio_aligned[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals