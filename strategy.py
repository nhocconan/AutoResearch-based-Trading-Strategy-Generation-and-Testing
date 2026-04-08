#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: On 1d timeframe, use KAMA for trend direction, RSI for momentum, and Choppiness Index for regime filter.
# Long when KAMA upward, RSI > 50, and Choppiness Index < 38.2 (trending market).
# Short when KAMA downward, RSI < 50, and Choppiness Index < 38.2.
# Exit when RSI crosses back to 50 or Choppiness Index > 61.8 (choppy market).
# Uses 1w timeframe for trend confirmation (EMA crossover) to avoid counter-trend trades.
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA, RSI, and Choppiness Index (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly: sum of absolute changes over ER period
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    change_t = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.zeros_like(close_1d)
    abs_change[1:] = np.abs(np.diff(close_1d))
    
    er = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        directional_change = np.abs(close_1d[i] - close_1d[i - er_period])
        total_change = np.sum(abs_change[i - er_period + 1:i + 1])
        if total_change > 0:
            er[i] = directional_change / total_change
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1))**2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    
    for i in range(rsi_period+1, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14)
    chop_period = 14
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(np.subtract(high, low))
    tr2 = np.abs(np.subtract(high, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low, np.roll(close_1d, 1)))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    for i in range(chop_period, len(close_1d)):
        atr[i] = np.mean(tr[i - chop_period + 1:i + 1])
    
    max_high = np.zeros_like(close_1d)
    min_low = np.zeros_like(close_1d)
    for i in range(chop_period, len(close_1d)):
        max_high[i] = np.max(high[i - chop_period + 1:i + 1])
        min_low[i] = np.min(low[i - chop_period + 1:i + 1])
    
    chop = np.zeros_like(close_1d)
    for i in range(chop_period, len(close_1d)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(np.sum(tr[i - chop_period + 1:i + 1]) / (max_high[i] - min_low[i])) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Get 1w data for trend confirmation (EMA crossover)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_fast = pd.Series(close_1w).ewm(span=10, adjust=False).mean().values
    ema_slow = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    
    # Align all indicators to 1d timeframe
    kama_1d = align_htf_to_ltf(prices, df_1d, kama)
    rsi_1d = align_htf_to_ltf(prices, df_1d, rsi)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop)
    ema_fast_1d = align_htf_to_ltf(prices, df_1w, ema_fast)
    ema_slow_1d = align_htf_to_ltf(prices, df_1w, ema_slow)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, rsi_period, chop_period)  # ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i]) or
            np.isnan(ema_fast_1d[i]) or np.isnan(ema_slow_1d[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or chop > 61.8 (choppy) or fast EMA < slow EMA (trend change)
            if (rsi_1d[i] < 50 or chop_1d[i] > 61.8 or ema_fast_1d[i] < ema_slow_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or chop > 61.8 (choppy) or fast EMA > slow EMA (trend change)
            if (rsi_1d[i] > 50 or chop_1d[i] > 61.8 or ema_fast_1d[i] > ema_slow_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: KAMA upward, RSI > 50, chop < 38.2 (trending), and fast EMA > slow EMA
            if (close[i] > kama_1d[i] and rsi_1d[i] > 50 and chop_1d[i] < 38.2 and ema_fast_1d[i] > ema_slow_1d[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA downward, RSI < 50, chop < 38.2 (trending), and fast EMA < slow EMA
            elif (close[i] < kama_1d[i] and rsi_1d[i] < 50 and chop_1d[i] < 38.2 and ema_fast_1d[i] < ema_slow_1d[i]):
                position = -1
                signals[i] = -0.25
    
    return signals