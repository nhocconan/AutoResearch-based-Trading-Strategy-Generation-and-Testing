#!/usr/bin/env python3
name = "1d_1w_KAMA_RSI_Trend_Follow"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA on 1w close
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    direction = np.abs(np.diff(close_1w, n=10, prepend=close_1w[:10]))
    er = np.where(change != 0, direction / change, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    kama_1w = kama
    
    # Align KAMA to 1d
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # RSI(14) on 1d close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA (uptrend) and RSI > 50 with volume
            if close[i] > kama_1w_aligned[i] and rsi[i] > 50 and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA (downtrend) and RSI < 50 with volume
            elif close[i] < kama_1w_aligned[i] and rsi[i] < 50 and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < KAMA or RSI < 40
            if close[i] < kama_1w_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > KAMA or RSI > 60
            if close[i] > kama_1w_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d trend following using 1w KAMA for trend filter and 1d RSI for entry/exit with volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in choppy markets. RSI >50/<50 confirms momentum direction.
# Volume filter ensures institutional participation. Works in both bull (trend following) and bear (avoids false breaks).
# Position size 0.25 limits drawdown. Target: ~15-25 trades/year.