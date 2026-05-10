#!/usr/bin/env python3
# 12h_KAMA_With_1wTrend_Filter_v2
# Hypothesis: KAMA trend direction on 12h combined with weekly trend filter (1w EMA50) and volume confirmation (1.5x 24-period average).
# Uses 12h timeframe for institutional entries, works in bull/bear markets via dual timeframe trend alignment.
# Target: 12-37 trades/year to minimize fee drag on 12h timeframe.

name = "12h_KAMA_With_1wTrend_Filter_v2"
timeframe = "12h"
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
    
    # 1w trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # KAMA calculation (ER=10) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_12h, n=10))
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)
    er = np.zeros_like(close_12h)
    er[10:] = change[10:] / np.where(volatility[10:] == 0, 1, volatility[10:])
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    # Trend direction
    kama_up = close_12h > kama
    kama_down = close_12h < kama
    
    # Align KAMA trend to 12h (itself)
    kama_up_aligned = align_htf_to_ltf(prices, df_12h, kama_up.astype(float))
    kama_down_aligned = align_htf_to_ltf(prices, df_12h, kama_down.astype(float))
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up with 1w uptrend and volume confirmation
            if (kama_up_aligned[i] > 0.5 and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down with 1w downtrend and volume confirmation
            elif (kama_down_aligned[i] > 0.5 and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA down or 1w trend turns down
            if (kama_down_aligned[i] > 0.5 or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA up or 1w trend turns up
            if (kama_up_aligned[i] > 0.5 or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals