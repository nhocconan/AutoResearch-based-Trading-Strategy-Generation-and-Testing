#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === Daily: KAMA trend direction ===
    close = prices['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close, k=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Vectorized calculation for ER
    er = np.zeros_like(close)
    for i in range(10, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Daily: RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === Weekly: Choppiness Index (CHOP) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = np.zeros(len(high_1w))
    for i in range(1, len(high_1w)):
        tr = max(high_1w[i] - low_1w[i],
                 abs(high_1w[i] - close_1w[i-1]),
                 abs(low_1w[i] - close_1w[i-1]))
        atr_1w[i] = (atr_1w[i-1] * 13 + tr) / 14 if i > 1 else tr
    # Sum of true ranges over 14 periods
    sum_tr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14)
    
    # Align weekly CHOP to daily
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop_aligned[i]
        close_val = close[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(chop_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending market)
            if (close_val > kama_val and
                rsi_val > 50 and
                chop_val < 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, chop < 61.8 (trending market)
            elif (close_val < kama_val and
                  rsi_val < 50 and
                  chop_val < 61.8):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down or RSI < 40
            if close_val < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up or RSI > 60
            if close_val > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals