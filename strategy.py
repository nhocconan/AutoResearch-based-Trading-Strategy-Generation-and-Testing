#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Chop
Hypothesis: On 12h timeframe, use KAMA for trend direction, RSI for overbought/oversold, and Choppiness Index for regime filtering. 
Long when KAMA rising, RSI < 40 (oversold), and choppy market (CHOP > 61.8). 
Short when KAMA falling, RSI > 60 (overbought), and choppy market (CHOP > 61.8).
Exit when RSI crosses 50. Designed for mean reversion in choppy markets with trend filter to avoid false signals.
Expected 50-150 trades over 4 years to minimize fee drag.
"""
name = "12h_KAMA_Direction_RSI_Chop"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else np.zeros_like(close)
        # ER calculation
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_vals = kama(close, 10, 2, 30)
    
    # Calculate RSI
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_vals = rsi(close, 14)
    
    # Calculate Choppiness Index
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False).mean().values
        
        max_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        
        chop_out = 100 * np.log10((atr * length) / (max_high - min_low + 1e-10)) / np.log10(length)
        return chop_out
    
    chop_vals = chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure sufficient warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or np.isnan(chop_vals[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: KAMA rising (current > previous), RSI oversold (<40), choppy market (CHOP > 61.8)
            if (kama_vals[i] > kama_vals[i-1] and 
                rsi_vals[i] < 40 and 
                chop_vals[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (current < previous), RSI overbought (>60), choppy market (CHOP > 61.8)
            elif (kama_vals[i] < kama_vals[i-1] and 
                  rsi_vals[i] > 60 and 
                  chop_vals[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: RSI crosses 50 (mean reversion complete)
            if position == 1:
                if rsi_vals[i] >= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if rsi_vals[i] <= 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals