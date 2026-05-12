#!/usr/bin/env python3
# 1D_KAMA_DIRECTION_RSI_CONFIRMATION
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) on 1d shows adaptive trend direction, confirmed by RSI(14) for momentum.
# In bull markets: KAMA rising + RSI > 50 signals long. In bear markets: KAMA falling + RSI < 50 signals short.
# Uses 1w trend filter to avoid counter-trend trades. Designed for low trade frequency (10-30/year) to minimize fee drag.
# Works in both bull and bear by following adaptive trend with momentum confirmation.

name = "1D_KAMA_DIRECTION_RSI_CONFIRMATION"
timeframe = "1d"
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
    
    # 1w data for trend filter (only use confirmed weekly closes)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1d KAMA calculation (adaptive trend)
    # Efficiency ratio over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI(14) for momentum confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length (first 14 values are NaN)
    rsi_padded = np.full_like(close, np.nan)
    rsi_padded[14:] = rsi
    
    # 1w EMA(34) for trend filter (only long in uptrend, short in downtrend)
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d and 1w data to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi_padded)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(15, n):  # Start after RSI warmup
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: KAMA rising (trend up) + RSI > 50 (bullish momentum) + 1w uptrend + volume confirmation
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and 
                close[i] > ema34_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling (trend down) + RSI < 50 (bearish momentum) + 1w downtrend + volume confirmation
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA turns down or RSI < 50
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA turns up or RSI > 50
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals