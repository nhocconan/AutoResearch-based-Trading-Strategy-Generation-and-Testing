#!/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_Filter_v2"
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
    
    # KAMA parameters
    er_len = 10
    fast = 2
    slow = 30
    
    # Direction (change) for ER
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
    # Avoid division by zero
    er = np.where(vol != 0, change / vol, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama_up = close > kama
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppy market filter: Chop > 61.8 = range (mean revert)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(np.roll(high, 1) - np.roll(close, 1))
    tr3 = np.abs(np.roll(low, 1) - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/atr_period, min_periods=atr_period).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.where(atr_sum != 0, 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(14), 50)
    chop_filter = chop > 61.8  # Range regime
    
    # Volume confirmation: volume > 1.2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.2 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI < 40 (oversold) + chop filter + volume
            if kama_up[i] and rsi[i] < 40 and chop_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI > 60 (overbought) + chop filter + volume
            elif not kama_up[i] and rsi[i] > 60 and chop_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA down OR RSI > 70
            if not kama_up[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA up OR RSI < 30
            if kama_up[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals