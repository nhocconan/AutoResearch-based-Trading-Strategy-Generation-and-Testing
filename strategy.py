#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_v1
Hypothesis: On daily timeframe, Kaufman Adaptive Moving Average (KAMA) determines trend direction.
RSI(14) provides entry timing: long when RSI crosses above 50 in uptrend, short when crosses below 50 in downtrend.
Choppiness Index (CHOP) regime filter: only trade when CHOP < 50 (trending market) to avoid whipsaws in ranging markets.
Volume confirmation: require volume > 1.5x 20-day average to ensure breakout conviction.
Designed for low trade frequency (<25/year) to minimize fee drag. Works in both bull and bear markets
by aligning with adaptive trend and avoiding false signals in choppy regimes.
"""

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
    volume = prices['volume'].values
    
    # Calculate 1d KAMA (adaptive trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=1)  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[29] = close_1d[29]  # seed
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    if len(df_1d) < 15:
        return np.zeros(n)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])  # align length
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Choppiness Index (CHOP)
    if len(df_1d) < 15:
        return np.zeros(n)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close_1d[:-1])
    tr3 = np.abs(low[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    # Max(high) - Min(low) over 14 periods
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_maxmin = max_high - min_low
    # CHOP = 100 * log10(sum(ATR) / (maxH - minL)) / log10(14)
    chop = 100 * np.log10(sum_atr / range_maxmin) / np.log10(14)
    chop = np.where(range_maxmin > 0, chop, 50)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 1.5 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        # Regime filter: only trade when market is trending (CHOP < 50)
        if chop_val >= 50:
            # In choppy regime, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Flat - look for entry: RSI crossover in direction of KAMA trend with volume spike
            # Long: RSI crosses above 50 AND price > KAMA (uptrend) AND volume spike
            # Short: RSI crosses below 50 AND price < KAMA (downtrend) AND volume spike
            if i > start_idx:
                rsi_prev = rsi_aligned[i-1]
                price_above_kama = close_val > kama_val
                price_below_kama = close_val < kama_val
                
                if (rsi_prev <= 50 and rsi_val > 50 and price_above_kama and vol_spike):
                    signals[i] = size
                    position = 1
                    entry_price = close_val
                elif (rsi_prev >= 50 and rsi_val < 50 and price_below_kama and vol_spike):
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long - exit when RSI crosses below 50 or trend changes
            if rsi_val < 50 or close_val < kama_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when RSI crosses above 50 or trend changes
            if rsi_val > 50 or close_val > kama_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0