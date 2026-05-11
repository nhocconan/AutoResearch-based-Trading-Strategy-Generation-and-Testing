#!/usr/bin/env python3
"""
6h_RSI2_WickReversal_1dTrend_Volume
Hypothesis: On 6h timeframe, use 2-period RSI for mean reversion with wick rejection signals, filtered by daily trend and volume spikes. RSI2 identifies extreme short-term overbought/oversold conditions, while long wicks indicate rejection of higher/lower prices. Daily trend ensures alignment with higher timeframe momentum, and volume spike confirms conviction. Designed for low trade frequency (15-25/year) to minimize fee drag in choppy 2025 markets.
"""

name = "6h_RSI2_WickReversal_1dTrend_Volume"
timeframe = "6h"
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
    
    # === Daily Trend Filter (EMA50) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Filter (2.0x 24-period EMA on 6h) ===
    vol_ema24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_ok = volume > vol_ema24 * 2.0
    
    # === RSI(2) on 6h close ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    def rma(src, length):
        result = np.full_like(src, np.nan, dtype=np.float64)
        alpha = 1.0 / length
        result[length-1] = np.mean(src[:length])
        for i in range(length, len(src)):
            result[i] = (src[i] * alpha) + (result[i-1] * (1 - alpha))
        return result
    
    rsi_period = 2
    avg_gain = rma(gain, rsi_period)
    avg_loss = rma(loss, rsi_period)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Wick Reversal Signals ===
    # Bullish: long lower wick (close near high, low far below)
    body_size = np.abs(close - open_)
    lower_wick = np.minimum(open_, close) - low
    upper_wick = high - np.maximum(open_, close)
    
    # Need open prices
    open_ = prices['open'].values
    
    # Avoid division by zero
    body_safe = np.where(body_size == 0, 1e-10, body_size)
    
    # Long lower wick ratio > 2.0 (wick at least 2x body)
    long_lower_wick = lower_wick > (2.0 * body_safe)
    # Long upper wick ratio > 2.0
    long_upper_wick = upper_wick > (2.0 * body_safe)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_6h[i]) or np.isnan(rsi[i]) or np.isnan(volume_ok[i]) or 
            np.isnan(long_lower_wick[i]) or np.isnan(long_upper_wick[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: RSI2 < 10 (extreme oversold) + long lower wick (rejection of lower prices)
            # Only in uptrend (price > daily EMA50)
            if (rsi[i] < 10 and 
                long_lower_wick[i] and 
                close[i] > ema50_6h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: RSI2 > 90 (extreme overbought) + long upper wick (rejection of higher prices)
            # Only in downtrend (price < daily EMA50)
            elif (rsi[i] > 90 and 
                  long_upper_wick[i] and 
                  close[i] < ema50_6h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI2 > 50 (mean reversion) or wick fails
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: RSI2 < 50 (mean reversion) or wick fails
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals