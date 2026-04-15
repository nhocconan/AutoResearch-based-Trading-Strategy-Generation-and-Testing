#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + Chop filter
# Uses 1d KAMA to identify trend direction, RSI(14) for momentum exhaustion,
# and 1d Choppiness Index to avoid ranging markets. Works in both bull and bear by
# taking long when KAMA rising + RSI < 30 (oversold) in trending markets,
# and short when KAMA falling + RSI > 70 (overbought) in trending markets.
# Target: 30-100 total trades over 4 years (7-25/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for KAMA, RSI and Chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (10-period ER, 2 and 30 for SC) on 1d
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14-period) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period) on 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    # Align all indicators to 1d timeframe (same as prices)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i])):
            continue
        
        # Long entry: KAMA rising + RSI < 30 (oversold) + chop < 61.8 (trending)
        if (kama_aligned[i] > kama_aligned[i-1] and
            rsi_aligned[i] < 30 and
            chop_aligned[i] < 61.8 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: KAMA falling + RSI > 70 (overbought) + chop < 61.8 (trending)
        elif (kama_aligned[i] < kama_aligned[i-1] and
              rsi_aligned[i] > 70 and
              chop_aligned[i] < 61.8 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: KAMA direction change or chop > 61.8 (ranging market)
        elif position == 1 and (kama_aligned[i] < kama_aligned[i-1] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (kama_aligned[i] > kama_aligned[i-1] or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0