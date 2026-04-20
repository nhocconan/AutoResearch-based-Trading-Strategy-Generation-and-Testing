#!/usr/bin/env python3
# Strategy: 12h_1d_KAMA_Direction_RSI_Filter
# Hypothesis: KAMA trend direction on 1d determines bias; RSI(2) overbought/oversold triggers mean-reversion entries on 12h. Works in both bull/bear by trading pullbacks in trend. Low trade frequency due to RSI extremes filter.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for KAMA trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA ( Kaufman Adaptive Moving Average ) - simplified with ER and smoothing constants
    close_1d_series = pd.Series(close_1d)
    change = abs(close_1d_series.diff(1))
    volatility = change.rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    sc = (er * (0.66 - 0.06) + 0.06) ** 2
    kama = [close_1d[0]]
    for i in range(1, len(close_1d)):
        kama.append(kama[-1] + sc.iloc[i] * (close_1d[i] - kama[-1]))
    kama = np.array(kama)
    kama_dir = np.where(close_1d > kama, 1, -1)  # 1: uptrend, -1: downtrend
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_dir)
    
    # RSI(2) on 1d for mean-reversion signals
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/2, min_periods=2, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi.values)
    
    # Load 12h data for entry execution
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if np.isnan(kama_dir_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        
        if position == 0:
            # Long: uptrend bias + RSI oversold
            if kama_dir_aligned[i] == 1 and rsi_aligned[i] < 15:
                signals[i] = 0.25
                position = 1
            # Short: downtrend bias + RSI overbought
            elif kama_dir_aligned[i] == -1 and rsi_aligned[i] > 85:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or trend change
            if rsi_aligned[i] > 70 or kama_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or trend change
            if rsi_aligned[i] < 30 or kama_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_KAMA_Direction_RSI_Filter"
timeframe = "12h"
leverage = 1.0