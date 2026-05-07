#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with RSI(14) pullback and volume confirmation.
# Uses Kaufman's Adaptive Moving Average to identify trend direction, enters on RSI pullbacks in trend direction,
# filtered by volume spikes. Designed for 12h timeframe to minimize trade frequency and avoid fee drag.
# Works in both bull and bear markets by following KAMA trend direction.
# Target: 12-30 trades/year per symbol to stay within optimal range.
name = "12h_KAMA_RSI_Pullback_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman's Adaptive Moving Average)
    # ER (Efficiency Ratio) = |change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    er = np.zeros_like(change)
    for i in range(10, len(change)):
        sum_change = np.sum(change[i-9:i+1])
        if sum_change > 0:
            er[i] = change[i] / sum_change
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection (20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and RSI
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from KAMA
        uptrend = close[i] > kama[i]
        downtrend = close[i] < kama[i]
        
        if position == 0:
            # Long: RSI pullback (<40) in uptrend with volume spike
            long_condition = (rsi[i] < 40) and uptrend and vol_spike[i]
            # Short: RSI pullback (>60) in downtrend with volume spike
            short_condition = (rsi[i] > 60) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 60 or trend turns down
            if (rsi[i] > 60) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 40 or trend turns up
            if (rsi[i] < 40) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals