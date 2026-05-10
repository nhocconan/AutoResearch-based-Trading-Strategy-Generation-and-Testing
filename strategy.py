#!/usr/bin/env python3
"""
4h_TRIX_ZeroLine_12hTrend_Volume
Hypothesis: TRIX (Triple Exponential Average) zero line cross indicates momentum shifts.
In trending markets, TRIX stays above/below zero; in ranging markets, it oscillates near zero.
We use 12h EMA50 as trend filter to ensure we trade only in the direction of higher timeframe trend.
Volume confirmation filters weak breakouts. Works in bull (TRIX>0 + uptrend) and bear (TRIX<0 + downtrend).
Target: 75-200 total trades over 4 years (19-50/year).
"""

name = "4h_TRIX_ZeroLine_12hTrend_Volume"
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
    volume = prices['volume'].values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume SMA20 for volume confirmation
    volume_12h = df_12h['volume'].values
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    # TRIX calculation (15-period as standard)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percentage change
    period = 15
    ema1 = np.full(n, np.nan)
    ema2 = np.full(n, np.nan)
    ema3 = np.full(n, np.nan)
    if n >= period:
        # First EMA
        ema1[period-1] = np.mean(close[:period])
        alpha = 2 / (period + 1)
        for i in range(period, n):
            ema1[i] = alpha * close[i] + (1 - alpha) * ema1[i-1]
        # Second EMA
        ema2[period-1] = np.mean(ema1[:period])
        for i in range(period, n):
            ema2[i] = alpha * ema1[i] + (1 - alpha) * ema2[i-1]
        # Third EMA
        ema3[period-1] = np.mean(ema2[:period])
        for i in range(period, n):
            ema3[i] = alpha * ema2[i] + (1 - alpha) * ema3[i-1]
    
    # TRIX = percentage change of triple EMA
    trix = np.full(n, np.nan)
    if n >= period + 1:
        for i in range(period, n):
            if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
                trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(period + 1, 50)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(trix[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 12h volume (scaled)
        vol_4h_approx = vol_sma20_12h_aligned[i] / 3.0  # 12h/4h = 3
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: TRIX crosses above zero with uptrend and volume
            if trix[i] > 0 and trix[i-1] <= 0 and close[i] > ema50_12h_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with downtrend and volume
            elif trix[i] < 0 and trix[i-1] >= 0 and close[i] < ema50_12h_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below zero or trend reversal
            if trix[i] < 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above zero or trend reversal
            if trix[i] > 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals