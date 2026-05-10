#!/usr/bin/env python3
# 12h_KAMA_With_1wTrend_Filter
# Hypothesis: KAMA direction (trend) filtered by 1w trend (EMA13) on 12h timeframe.
# Uses adaptive trend strength to avoid whipsaws in sideways markets.
# Target: 20-40 trades/year to minimize fee drag on 12h timeframe.

name = "12h_KAMA_With_1wTrend_Filter"
timeframe = "12h"
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
    
    # 1w trend filter (EMA13)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    trend_1w_up = close_1w > ema13_1w
    trend_1w_down = close_1w < ema13_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # KAMA direction (ER = Efficiency Ratio, smooth with 2 and 30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))).cumsum()  # placeholder, will fix below
    # Recalculate volatility properly as rolling sum of absolute changes
    volatility = np.zeros(n)
    for i in range(n):
        if i == 0:
            volatility[i] = 0
        else:
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
            if i >= 30:
                volatility[i] -= np.abs(close[i-30] - close[i-30-1]) if i-30-1 >= 0 else 0
    # Avoid division by zero
    volatility = np.where(volatility == 0, 1e-10, volatility)
    er = np.where(volatility > 0, change / volatility, 0)
    # Smooth ER with smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (ER * (fast - slow) + slow)^2
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # KAMA direction: price above/below KAMA
    kama_up = close > kama
    kama_down = close < kama
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(kama_up[i]) or np.isnan(kama_down[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume confirmation, 1w uptrend
            if (kama_up[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation, 1w downtrend
            elif (kama_down[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or 1w trend turns down
            if (not kama_up[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or 1w trend turns up
            if (not kama_down[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals