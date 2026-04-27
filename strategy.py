#!/usr/bin/env python3
"""
Hypothesis: 1d RSI divergence with volume confirmation and 1w trend filter.
- RSI divergence (bullish/bearish) signals potential reversals at extremes
- Volume spike confirms institutional participation during divergence
- 1w EMA50 trend filter ensures alignment with higher timeframe trend
- Target: 15-25 trades/year to minimize fee drag
- Uses discrete position sizing (0.25) to avoid churn
"""

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
    
    # Get daily data for RSI and volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on daily data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smooth(gain, 14)
    avg_loss = wilders_smooth(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate weekly EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        multiplier = 2 / (period + 1)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = (values[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema50_1w = ema(close_1w, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume spike: daily volume > 2.0 * 20-period average
    vol_ma_20 = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 100)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: price makes lower low, RSI makes higher low + volume spike + above weekly EMA50
            if (i >= 2 and 
                low[i] < low[i-1] and low[i-1] < low[i-2] and  # price lower low
                rsi_aligned[i] > rsi_aligned[i-1] and rsi_aligned[i-1] > rsi_aligned[i-2] and  # RSI higher low
                rsi_aligned[i] < 30 and  # oversold
                volume_spike_aligned[i] and 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Bearish divergence: price makes higher high, RSI makes lower high + volume spike + below weekly EMA50
            elif (i >= 2 and 
                  high[i] > high[i-1] and high[i-1] > high[i-2] and  # price higher high
                  rsi_aligned[i] < rsi_aligned[i-1] and rsi_aligned[i-1] < rsi_aligned[i-2] and  # RSI lower high
                  rsi_aligned[i] > 70 and  # overbought
                  volume_spike_aligned[i] and 
                  close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 70 (overbought) or price below weekly EMA50
            if (rsi_aligned[i] > 70 or 
                close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI < 30 (oversold) or price above weekly EMA50
            if (rsi_aligned[i] < 30 or 
                close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSIDivergence_VolumeSpike_1wEMA50_v1"
timeframe = "1d"
leverage = 1.0