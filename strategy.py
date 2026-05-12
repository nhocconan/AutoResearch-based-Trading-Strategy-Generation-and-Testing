#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Regime
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing reliable trend direction.
In trending markets (Choppiness Index < 38.2), we follow KAMA direction. In ranging markets (Choppiness Index > 61.8),
we fade moves at RSI extremes. Daily timeframe reduces trade frequency to avoid fee drag.
Works in bull/bear: adapts to trend strength via efficiency ratio and uses mean reversion in ranges.
"""

name = "1d_KAMA_RSI_Chop"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1-DAY DATA FOR INDICATORS (already matches timeframe, but using helper for consistency) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === KAMA (10-period ER, 2/30 fast/slow) ===
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # placeholder, will compute correctly below
    
    # Correct efficiency ratio calculation
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 1.0
        else:
            direction = np.abs(close_1d[i] - close_1d[0])
            volatility_sum = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1])))
            er[i] = direction / (volatility_sum + 1e-10) if volatility_sum > 0 else 1.0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === RSI (14-period) ===
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === CHOPPINESS INDEX (14-period) ===
    atr_list = []
    for i in range(len(high_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_list.append(tr)
    atr = pd.Series(atr_list).rolling(window=14, min_periods=14).mean().values
    
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(sum(atr_list[max(0, i-13):i+1]) / (max_high[i] - min_low[i])) / np.log10(14)
        else:
            chop[i] = 50.0
    
    # === ALIGN INDICATORS TO 1D TIMEFRAME (no actual alignment needed as timeframes match, but using helper for consistency) ===
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # For RSI, Chop, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # TREND MODE: Chop < 38.2 -> follow KAMA
            if chop_aligned[i] < 38.2:
                if close[i] > kama_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                elif close[i] < kama_aligned[i]:
                    signals[i] = -0.30
                    position = -1
            # RANGE MODE: Chop > 61.8 -> fade RSI extremes
            elif chop_aligned[i] > 61.8:
                if rsi_aligned[i] < 30 and close[i] > kama_aligned[i]:  # Oversold + price above KAMA
                    signals[i] = 0.30
                    position = 1
                elif rsi_aligned[i] > 70 and close[i] < kama_aligned[i]:  # Overbought + price below KAMA
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # EXIT LONG: Chop > 61.8 and RSI > 70 (overbought in range) OR close below KAMA in trend
            if chop_aligned[i] > 61.8 and rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] < 38.2 and close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Chop > 61.8 and RSI < 30 (oversold in range) OR close above KAMA in trend
            if chop_aligned[i] > 61.8 and rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] < 38.2 and close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals