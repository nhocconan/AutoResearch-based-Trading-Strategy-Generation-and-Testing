#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_kama_reversion_v1
# Uses 1-day KAMA to establish trend direction on 12h chart.
# Enters long when price crosses above KAMA with volume confirmation in downtrend (mean reversion).
# Enters short when price crosses below KAMA with volume confirmation in uptrend (mean reversion).
# Uses 1-day RSI to filter extremes (RSI < 30 for long, RSI > 70 for short).
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
# Works in ranging markets (mean reversion) and avoids strong trends via RSI filter.

name = "12h_1d_kama_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # ER (Efficiency Ratio) = |change| / sum(|abs changes|)
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    price_change = np.abs(np.diff(close_1d, 10))
    er = np.where(volatility != 0, price_change / volatility, 0)
    # Smooth ER with smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start with first value
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    # Wilder's smoothing for first average
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for alignment (since we lose first element in diff)
    rsi = np.concatenate([[np.nan], rsi])
    
    # Align KAMA and RSI to 12h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: volume > 1.5 * 20-period average (on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if values not ready
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price crosses above KAMA AND RSI < 30 (oversold)
        if close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1] and rsi_aligned[i] < 30:
            if position != 1:
                position = 1
                signals[i] = 0.25
            else:
                signals[i] = 0.25  # maintain position
        # Short signal: price crosses below KAMA AND RSI > 70 (overbought)
        elif close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1] and rsi_aligned[i] > 70:
            if position != -1:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = -0.25  # maintain position
        # Exit conditions: opposite cross OR RSI returns to neutral zone (40-60)
        elif (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) or \
             (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) or \
             (40 <= rsi_aligned[i] <= 60):
            if position == 1:
                position = 0
                signals[i] = 0.0
            elif position == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals