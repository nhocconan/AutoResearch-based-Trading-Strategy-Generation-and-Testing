#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + chop regime for long-only trend following
# KAMA adapts to market noise - follows trend in low volatility, stays flat in chop
# RSI(14) > 50 confirms bullish momentum
# Choppiness Index (CHOP) > 61.8 defines ranging market (avoid trades)
# Long when KAMA trend up AND RSI > 50 AND CHOP <= 61.8
# Flat otherwise
# Designed for 1d timeframe to target 10-25 trades/year per symbol.
# Works in bull markets by following trends, avoids whipsaws in bear/chop via KAMA and chop filter.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data for chop regime filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # KAMA(10) for trend - adaptive moving average
    # Efficiency Ratio: ER = |net change| / sum(|abs change|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # Calculate ER over 10 periods
    net_change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    total_change = np.sum(np.abs(np.diff(close, n=1, prepend=close[:1])).reshape(-1, 1)[:, :10], axis=1)
    # Simpler approach: calculate ER directly
    er = np.zeros(n)
    for i in range(10, n):
        net = np.abs(close[i] - close[i-10])
        total = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if total > 0:
            er[i] = net / total
        else:
            er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14) on weekly data
    atr_1w = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        tr = max(
            high_1w[i] - low_1w[i],
            np.abs(high_1w[i] - close_1w[i-1]),
            np.abs(low_1w[i] - close_1w[i-1])
        )
        atr_1w[i] = tr
    
    # Smooth ATR
    atr_ma_1w = np.zeros(len(close_1w))
    for i in range(14, len(close_1w)):
        atr_ma_1w[i] = np.mean(atr_1w[i-13:i+1])
    
    # Calculate Chop
    sum_atr_1w = np.zeros(len(close_1w))
    for i in range(14, len(close_1w)):
        sum_atr_1w[i] = np.sum(atr_ma_1w[i-13:i+1])
    
    max_high_1w = np.zeros(len(close_1w))
    min_low_1w = np.zeros(len(close_1w))
    for i in range(14, len(close_1w)):
        max_high_1w[i] = np.max(high_1w[i-13:i+1])
        min_low_1w[i] = np.min(low_1w[i-13:i+1])
    
    chop = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if max_high_1w[i] > min_low_1w[i]:
            chop[i] = 100 * np.log10(sum_atr_1w[i] / (max_high_1w[i] - min_low_1w[i])) / np.log10(14)
        else:
            chop[i] = 50
    
    # Align Chop to daily timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            continue
        
        # Long-only: KAMA up (price > KAMA) AND RSI > 50 AND not choppy (CHOP <= 61.8)
        if (close[i] > kama[i] and 
            rsi[i] > 50 and 
            chop_aligned[i] <= 61.8):
            signals[i] = 0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_LongOnly"
timeframe = "1d"
leverage = 1.0