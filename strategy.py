#!/usr/bin/env python3
# 1d_KAMA_With_1wTrend_Filter_v2
# Hypothesis: Daily KAMA trend filter with weekly EMA50 trend alignment and volume confirmation.
# Uses KAMA to capture adaptive trend with lower whipsaw in sideways markets.
# Weekly EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume confirmation filters low-momentum breakouts.
# Target: 10-25 trades/year to minimize fee drag on daily timeframe.

name = "1d_KAMA_With_1wTrend_Filter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None
    # Recalculate volatility properly: sum of absolute changes over ER period
    er = np.zeros(n)
    for i in range(n):
        if i < 10:
            er[i] = np.nan
        else:
            direction = np.abs(close[i] - close[i-10])
            volatility_sum = 0
            for j in range(1, 11):
                volatility_sum += np.abs(close[i-j+1] - close[i-j])
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation (1.5x 20-day average)
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
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA with weekly uptrend and volume confirmation
            if (close[i] > kama[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with weekly downtrend and volume confirmation
            elif (close[i] < kama[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below KAMA or weekly trend turns down
            if (close[i] < kama[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above KAMA or weekly trend turns up
            if (close[i] > kama[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals