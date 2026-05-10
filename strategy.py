#!/usr/bin/env python3
# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: KAMA adapts to market noise, capturing trends while avoiding whipsaws.
# Combined with RSI for momentum and Choppiness index for regime detection, this strategy
# aims to capture strong trends in both bull and bear markets while avoiding chop.
# Uses weekly trend filter for higher timeframe confirmation.
# Target: 10-25 trades/year to minimize fee drag on 1d timeframe.

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum() - np.abs(np.diff(close, prepend=close[0])).cumsum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_val = kama(close, 10, 2, 30)
    kama_dir = np.zeros_like(close)
    kama_dir[1:] = np.where(kama_val[1:] > kama_val[:-1], 1, -1)
    
    # Calculate RSI (Relative Strength Index)
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[length] = np.mean(gain[1:length+1])
        avg_loss[length] = np.mean(loss[1:length+1])
        for i in range(length+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_val = rsi(close, 14)
    
    # Calculate Choppiness Index
    def choppiness_index(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            if i == 1:
                atr[i] = tr[i]
            else:
                atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        sum_atr = np.zeros_like(close)
        for i in range(length, len(close)):
            sum_atr[i] = np.sum(atr[i-length+1:i+1])
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(length, len(close)):
            max_high[i] = np.max(high[i-length+1:i+1])
            min_low[i] = np.min(low[i-length+1:i+1])
        chop = np.zeros_like(close)
        for i in range(length, len(close)):
            if sum_atr[i] != 0 and (max_high[i] - min_low[i]) != 0:
                chop[i] = 100 * np.log10(sum_atr[i] / (max_high[i] - min_low[i])) / np.log10(length)
        return chop
    
    chop_val = choppiness_index(high, low, close, 14)
    
    # Weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1w_up = close_1w > ema34_1w
    trend_1w_down = close_1w < ema34_1w
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_val[i]) or np.isnan(chop_val[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, Chop < 61.8 (trending), weekly uptrend
            if (kama_dir[i] > 0 and
                rsi_val[i] > 50 and
                chop_val[i] < 61.8 and
                trend_1w_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, Chop < 61.8 (trending), weekly downtrend
            elif (kama_dir[i] < 0 and
                  rsi_val[i] < 50 and
                  chop_val[i] < 61.8 and
                  trend_1w_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA down or RSI < 40 or Chop > 61.8 (choppy) or weekly trend turns down
            if (kama_dir[i] < 0 or
                rsi_val[i] < 40 or
                chop_val[i] > 61.8 or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA up or RSI > 60 or Chop > 61.8 (choppy) or weekly trend turns up
            if (kama_dir[i] > 0 or
                rsi_val[i] > 60 or
                chop_val[i] > 61.8 or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals