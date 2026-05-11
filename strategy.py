#!/usr/bin/env python3
name = "4h_KAMA_34_RSI_20_Chop_30"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # KAMA calculation (ER = Efficiency Ratio)
    change_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) > 1 else np.array([0.0])
    # Use rolling window for volatility
    volatility_roll = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=10).sum().values
    change_roll = np.abs(close_1d - np.roll(close_1d, 10))
    change_roll[:10] = 0
    er = np.where(volatility_roll != 0, change_roll / volatility_roll, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # kama with fast=2, slow=30
    sc[:30] = 0  # ensure warmup
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI calculation on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index on 1d (using high/low/close)
    atr_1d = np.maximum(np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1))), np.abs(low_1d - np.roll(close_1d, 1)))
    atr_1d[0] = high_1d[0] - low_1d[0]
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) != 0, chop, 50)
    
    # Align all 1d indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike: current volume > 2x 20-period average (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * volume_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > KAMA, RSI > 50, Chop < 30 (trending), Volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] > 50 and
                chop_aligned[i] < 30 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA, RSI < 50, Chop < 30 (trending), Volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] < 50 and
                  chop_aligned[i] < 30 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below KAMA OR Chop > 60 (choppy)
            if close[i] < kama_aligned[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above KAMA OR Chop > 60 (choppy)
            if close[i] > kama_aligned[i] or chop_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals