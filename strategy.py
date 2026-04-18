# 1d_KAMA_RSI_Chop_Filter
# Hypothesis: 1d KAMA trend filter + RSI extremes + Chop regime to reduce false signals in low volatility.
# Works in bull/bear via trend filter, reduces whipsaw with chop filter, limits trades via strict entry.
# Target: 15-25 trades/year to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Chop filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on weekly for Chop denominator
    tr_1w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[:14])
        else:
            atr_1w[i] = (tr_1w[i] * 1/14) + (atr_1w[i-1] * 13/14)
    
    # Calculate Chop(14) on weekly: 100 * log10(sum(TR(14)) / (max(HH) - min(LL))) / log10(14)
    chop_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        sum_tr = np.sum(tr_1w[i-13:i+1])  # 14-period sum
        max_hh = np.max(high_1w[i-13:i+1])
        min_ll = np.min(low_1w[i-13:i+1])
        if max_hh != min_ll:
            chop_1w[i] = 100 * np.log10(sum_tr / (max_hh - min_ll)) / np.log10(14)
        else:
            chop_1w[i] = 50  # neutral if no range
    
    # Chop > 61.8 = range (mean revert), Chop < 38.2 = trending (trend follow)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Daily KAMA for trend
    # Efficiency Ratio: ER = |close - close(10)| / sum(|change|, 10)
    change = np.abs(np.diff(close))
    change = np.insert(change, 0, 0)  # align length
    abs_change_sum = np.full(n, np.nan)
    for i in range(10, n):
        if i == 10:
            abs_change_sum[i] = np.sum(change[1:11])
        else:
            abs_change_sum[i] = abs_change_sum[i-1] - change[i-10] + change[i]
    price_change = np.abs(close - np.roll(close, 10))
    price_change[:10] = 0
    er = np.zeros(n)
    er[:] = np.where(abs_change_sum != 0, price_change / abs_change_sum, 0)
    # Smoothing constants: fastest EMA(2) = 2/(2+1)=0.667, slowest EMA(30)=2/(30+1)=0.0645
    sc = (er * (0.667 - 0.0645) + 0.0645) ** 2
    kama = np.full(n, np.nan)
    for i in range(10, n):
        if i == 10:
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI(14)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = np.full(n, np.nan)
    roll_down = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            roll_up[i] = np.mean(up[1:15])
            roll_down[i] = np.mean(down[1:15])
        else:
            roll_up[i] = (up[i] + 13 * roll_up[i-1]) / 14
            roll_down[i] = (down[i] + 13 * roll_down[i-1]) / 14
    rs = np.where(roll_down != 0, roll_up / roll_down, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need KAMA, RSI, Chop
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: Chop > 50 = range (mean revert), Chop < 50 = trending (trend follow)
        chop_range = chop_1w_aligned[i] > 50
        chop_trend = chop_1w_aligned[i] < 50
        
        if position == 0:
            # Long: KAMA up + RSI < 30 (oversold) in ranging OR RSI > 50 in trending
            if kama[i] > kama[i-1]:
                if (chop_range and rsi[i] < 30) or (chop_trend and rsi[i] > 50):
                    signals[i] = 0.25
                    position = 1
            # Short: KAMA down + RSI > 70 (overbought) in ranging OR RSI < 50 in trending
            elif kama[i] < kama[i-1]:
                if (chop_range and rsi[i] > 70) or (chop_trend and rsi[i] < 50):
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: KAMA down OR RSI > 70
            if kama[i] < kama[i-1] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up OR RSI < 30
            if kama[i] > kama[i-1] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0