#!/usr/bin/env python3
"""
4h_RSI_Overbought_Oversold_Trend_v1
RSI(14) > 70 for short, RSI(14) < 30 for long with EMA(50) trend filter on 4h.
Exit when RSI crosses back to neutral zone (40-60).
Designed to capture mean reversion in strong trends with momentum exhaustion.
Target: 50-150 total trades over 4 years (12-37/year).
"""

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
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === EMA(50) trend filter on 4h ===
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: RSI < 30 (oversold) and price above EMA50 (uptrend)
            if (rsi[i] < 30 and 
                close[i] > ema_50[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: RSI > 70 (overbought) and price below EMA50 (downtrend)
            elif (rsi[i] > 70 and 
                  close[i] < ema_50[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: RSI > 40 (exiting oversold)
            if rsi[i] > 40:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI < 60 (exiting overbought)
            if rsi[i] < 60:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_Overbought_Oversold_Trend_v1"
timeframe = "4h"
leverage = 1.0