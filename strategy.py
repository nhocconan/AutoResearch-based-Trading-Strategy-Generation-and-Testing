#!/usr/bin/env python3
# 12h_KAMA_Direction_RSI_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) tracks trend with low whipsaw. 
# Entry when KAMA direction aligns with RSI > 50 (bull) or < 50 (bear) on 12h timeframe.
# Uses 1d trend filter (EMA50) to avoid counter-trend trades. Volume confirmation (1.5x 24-period average) reduces false signals.
# Designed for 12h timeframe to target 12-37 trades/year, minimizing fee drag while capturing major trends.

name = "12h_KAMA_Direction_RSI_Filter"
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
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # KAMA on 12h close
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, k=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_12h, k=1)), axis=0)  # sum |close[t] - close[t-1]|
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.full_like(change, 0.1, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # start after 10 periods
    for i in range(10, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_dir = np.where(kama_dir > 0, 1, np.where(kama_dir < 0, -1, 0))
    
    # Align KAMA direction to 12h timeframe (already on 12h, but need to align to 12h index of prices)
    # Since df_12h is already 12h data, we need to map its index to prices index
    kama_dir_aligned = align_htf_to_ltf(prices, df_12h, kama_dir.astype(float))
    
    # RSI on 12h close (14-period)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 1.0, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume confirmation (1.5x 24-period average on 12h)
    # Note: volume is already at 12h frequency since prices is 12h
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
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, 1d uptrend, volume confirmation
            if (kama_dir_aligned[i] > 0 and
                rsi_aligned[i] > 50 and
                trend_1d_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, RSI < 50, 1d downtrend, volume confirmation
            elif (kama_dir_aligned[i] < 0 and
                  rsi_aligned[i] < 50 and
                  trend_1d_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA falls or RSI < 40 or 1d trend turns down
            if (kama_dir_aligned[i] < 0 or
                rsi_aligned[i] < 40 or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA rises or RSI > 60 or 1d trend turns up
            if (kama_dir_aligned[i] > 0 or
                rsi_aligned[i] > 60 or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals