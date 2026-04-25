#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter
Hypothesis: Use 1d timeframe with Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index for regime filtering. Enter long
when KAMA slope positive, RSI > 50, and CHOP < 38.2 (trending regime). Enter short when
KAMA slope negative, RSI < 50, and CHOP < 38.2. Exit on opposite conditions. Uses discrete
sizing 0.25 to limit drawdown. Target 20-50 trades/year on 1d timeframe. Works in bull/bear
via adaptive trend and regime filter.
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
    
    # Get 1w data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA(10) on 1d close
    # Efficiency ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.abs(np.diff(close, n=1))  # |close[t] - close[t-1]|
    
    # Pad change array to align with close
    change_padded = np.concatenate([np.full(10, np.nan), change])
    
    # Calculate rolling sum of volatility
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    vol_sum_padded = np.concatenate([np.full(10, np.nan), vol_sum])
    
    # Avoid division by zero
    er = np.where(vol_sum_padded != 0, change_padded / vol_sum_padded, 0)
    er = np.nan_to_num(er, nan=0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # Seed with first close
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.nan_to_num(rsi, nan=50)
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero and log of zero
    hh_ll = hh - ll
    chop_ratio = np.where(hh_ll > 0, tr_sum / hh_ll, 1)
    chop_ratio = np.maximum(chop_ratio, 1e-10)  # Avoid log(0)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50)
    
    # Align weekly EMA34 to 1d
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), Chop (14), EMA34 (34)
    start_idx = max(10, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA rising, RSI > 50, Chop < 38.2 (trending), price above weekly EMA34
            long_setup = (kama_slope[i] > 0) and \
                         (rsi[i] > 50) and \
                         (chop[i] < 38.2) and \
                         (close[i] > ema_34_1w_aligned[i])
            # Short: KAMA falling, RSI < 50, Chop < 38.2 (trending), price below weekly EMA34
            short_setup = (kama_slope[i] < 0) and \
                          (rsi[i] < 50) and \
                          (chop[i] < 38.2) and \
                          (close[i] < ema_34_1w_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA falling OR RSI < 40 OR Chop > 61.8 (ranging) OR price below weekly EMA34
            if (kama_slope[i] < 0) or \
               (rsi[i] < 40) or \
               (chop[i] > 61.8) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA rising OR RSI > 60 OR Chop > 61.8 (ranging) OR price above weekly EMA34
            if (kama_slope[i] > 0) or \
               (rsi[i] > 60) or \
               (chop[i] > 61.8) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0