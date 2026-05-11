#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_RSI_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise - in trending markets it follows price closely, in ranging markets it stays flat.
- Use daily KAMA to determine higher timeframe trend direction
- Enter long when price crosses above KAMA AND RSI(14) > 50 (bullish momentum)
- Enter short when price crosses below KAMA AND RSI(14) < 50 (bearish momentum)
- Exit when price crosses back across KAMA in opposite direction
KAMA reduces whipsaws in ranging markets while capturing trends. Works in bull by following uptrends, in bear by avoiding false breakdowns.
Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
"""

name = "4h_1d_KAMA_Direction_RSI_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- 1d KAMA: Kaufman Adaptive Moving Average ---
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else np.array([0])
    # Actually compute properly: ER = |change| / sum(|changes|) over period
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate ER for each point
    er = np.full_like(close_1d, np.nan)
    for i in range(er_period, len(close_1d)):
        if i >= er_period:
            direction = np.abs(close_1d[i] - close_1d[i - er_period])
            volatility_sum = np.sum(np.abs(np.diff(close_1d[i - er_period:i])))
            if volatility_sum > 0:
                er[i] = direction / volatility_sum
            else:
                er[i] = 0
    
    # Smoothing constant: SC = [ER * (fastest SC - slowest SC) + slowest SC]^2
    sc = np.full_like(close_1d, np.nan)
    for i in range(len(close_1d)):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        else:
            sc[i] = 0
    
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # --- RSI(14) on 4h ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    rsi_period = 14
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    
    # First average
    if len(gain) >= rsi_period:
        avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    # Wilder smoothing: avg = (prev_avg * (period-1) + current_value) / period
    for i in range(rsi_period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(30, rsi_period + 2)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for entries: price cross KAMA with RSI confirmation
            if close[i] > kama_aligned[i] and rsi[i] > 50:
                # Long: price above KAMA AND bullish momentum
                signals[i] = 0.25
                position = 1
            elif close[i] < kama_aligned[i] and rsi[i] < 50:
                # Short: price below KAMA AND bearish momentum
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses back across KAMA
            if position == 1:
                # Exit long: price crosses back below KAMA
                if close[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses back above KAMA
                if close[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals