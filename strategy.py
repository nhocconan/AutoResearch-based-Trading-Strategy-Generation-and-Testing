#/usr/bin/env python3
# 1d_1w_kama_rsi_v1
# Hypothesis: KAMA(14) on daily captures adaptive trend while RSI(14) on weekly filters extremes. Long when KAMA trending up and weekly RSI < 70; short when KAMA trending down and weekly RSI > 30. Designed to avoid whipsaws in ranging markets and catch trends in both bull and bear regimes. Low-frequency signals target 7-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # KAMA on daily (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    er[volatility != 0] = change[volatility != 0] / volatility[volatility != 0]
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly RSI(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR weekly RSI > 70 (overbought)
            if (kama[i] < kama[i-1]) or (rsi_1w_aligned[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR weekly RSI < 30 (oversold)
            if (kama[i] > kama[i-1]) or (rsi_1w_aligned[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: KAMA trending up AND weekly RSI not overbought
            if (kama[i] > kama[i-1]) and (rsi_1w_aligned[i] < 70):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA trending down AND weekly RSI not oversold
            elif (kama[i] < kama[i-1]) and (rsi_1w_aligned[i] > 30):
                position = -1
                signals[i] = -0.25
    
    return signals