#!/usr/bin/env python3
"""
1d_RSI_Extremes_TrendFollow_v1
Hypothesis: RSI extremes (<30 or >70) with trend confirmation (price > SMA200) capture reversal 
entries in trending markets. Works in both bull and bear by using trend filter to avoid counter-trend 
trades. Low frequency (~10-25 trades/year) via strict RSI thresholds and trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # SMA200 trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need RSI (14), SMA200 (200)
    start_idx = 200
    
    for i in range(start_idx, n):
        rsi_val = rsi[i]
        close_val = close[i]
        sma200_val = sma200[i]
        
        if np.isnan(rsi_val) or np.isnan(sma200_val):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > SMA200 (uptrend)
            if rsi_val < 30 and close_val > sma200_val:
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) and price < SMA200 (downtrend)
            elif rsi_val > 70 and close_val < sma200_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: RSI > 50 (momentum shift) or stop/reversal
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI < 50 (momentum shift) or stop/reversal
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_Extremes_TrendFollow_v1"
timeframe = "1d"
leverage = 1.0