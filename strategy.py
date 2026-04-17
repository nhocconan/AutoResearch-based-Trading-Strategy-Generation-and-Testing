#!/usr/bin/env python3
"""
12h_KAMA_Trend_1d_RSI_Momentum
- Uses KAMA (Kaufman Adaptive Moving Average) on 12h for adaptive trend detection
- Filters with 1d RSI to avoid extreme overbought/oversold conditions
- Enters long when KAMA slopes up and RSI < 70, short when KAMA slopes down and RSI > 30
- Exits when KAMA slope reverses or RSI reaches opposite extreme
- Position size: 0.25 for trend alignment, 0.0 otherwise
- Designed for 12h timeframe to reduce trade frequency and avoid fee drag
- Works in bull (trend following) and bear (mean reversion via RSI extremes) regimes
"""

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
    
    # Get 12h data for KAMA calculation
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA on 12h close
    close_12h = df_12h['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_12h, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2 / (2 + 1) - 2 / (30 + 1)) + 2 / (30 + 1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[29] = close_12h[29]  # seed
    for i in range(30, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    
    # Calculate RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align KAMA and RSI to 12h timeframe
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi, additional_delay_bars=0)  # RSI uses current 1d bar
    
    # Calculate KAMA slope (1-period change)
    kama_slope = np.diff(kama_12h_aligned, prepend=0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(kama_slope[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_12h_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        slope = kama_slope[i]
        
        if position == 0:
            # Long: KAMA slope up and RSI not overbought
            if slope > 0 and rsi_val < 70:
                signals[i] = 0.25
                position = 1
            # Short: KAMA slope down and RSI not oversold
            elif slope < 0 and rsi_val > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA slope down or RSI overbought
            if slope <= 0 or rsi_val >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA slope up or RSI oversold
            if slope >= 0 or rsi_val <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_1d_RSI_Momentum"
timeframe = "12h"
leverage = 1.0