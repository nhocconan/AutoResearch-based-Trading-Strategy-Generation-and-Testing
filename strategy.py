#!/usr/bin/env python3
"""
4h_GoldenCross_DeathCross_RSIFilter_v1
Concept: 4h EMA(50) crossing EMA(200) with RSI(14) filter to avoid false signals.
- Long: EMA50 crosses above EMA200 AND RSI < 70 (not overbought)
- Short: EMA50 crosses below EMA200 AND RSI > 30 (not oversold)
- Exit: Opposite cross OR RSI reaches extreme (80/20)
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Works in bull/bear: EMA crossover defines trend, RSI filter prevents entries during exhaustion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_GoldenCross_DeathCross_RSIFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 210:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # === 4h: EMA50 and EMA200 ===
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === 4h: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Get values
        ema50_now = ema50[i]
        ema50_prev = ema50[i-1]
        ema200_now = ema200[i]
        ema200_prev = ema200[i-1]
        rsi_now = rsi[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_now) or np.isnan(ema50_prev) or np.isnan(ema200_now) or 
            np.isnan(ema200_prev) or np.isnan(rsi_now)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Golden Cross: EMA50 crosses above EMA200 AND RSI not overbought
            if ema50_now > ema200_now and ema50_prev <= ema200_prev and rsi_now < 70:
                signals[i] = 0.25
                position = 1
            # Death Cross: EMA50 crosses below EMA200 AND RSI not oversold
            elif ema50_now < ema200_now and ema50_prev >= ema200_prev and rsi_now > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Death Cross OR RSI overbought
            if (ema50_now < ema200_now and ema50_prev >= ema200_prev) or rsi_now >= 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Golden Cross OR RSI oversold
            if (ema50_now > ema200_now and ema50_prev <= ema200_prev) or rsi_now <= 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals