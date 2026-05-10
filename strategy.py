#!/usr/bin/env python3
"""
4h_RSI2_OverboughtOversold_TrendFilter
Hypothesis: RSI(2) identifies short-term extremes in overbought/oversold conditions.
Combined with 1-day EMA50 trend filter and volume confirmation to avoid false signals.
Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
Target: 20-30 trades/year with strict entry conditions to minimize fee drag.
"""

name = "4h_RSI2_OverboughtOversold_TrendFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate RSI(2) on 4h closes
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    def rma(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    rsi_period = 2
    avg_gain = rma(gain, rsi_period)
    avg_loss = rma(loss, rsi_period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period EMA
    volume = prices['volume'].values
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need RSI(2) and EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend: price vs EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: uptrend AND RSI(2) < 10 (extremely oversold) with volume
            if uptrend and rsi[i] < 10 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND RSI(2) > 90 (extremely overbought) with volume
            elif downtrend and rsi[i] > 90 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI(2) > 50 (mean reversion) OR trend changes to downtrend
            if rsi[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI(2) < 50 (mean reversion) OR trend changes to uptrend
            if rsi[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals