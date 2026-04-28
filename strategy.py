#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_Volume_Spike
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 12h to capture trend direction, with volume spikes (2x 24-period average) to confirm momentum. Works in bull/bear by following trend direction only. Targets 15-25 trades/year via strict entry conditions. Uses 1w RSI as regime filter to avoid extreme overbought/oversold conditions.
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
    
    # Get 12h data for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1w data for RSI regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate KAMA (ER=10, fast=2, slow=30) on 12h
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h)), axis=0)
    # Fix: volatility should be rolling sum
    volatility = pd.Series(close_12h).diff().abs().rolling(window=30, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_12h)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
    kama_12h = kama
    
    # Align KAMA to 12h timeframe (already on 12h, but need to align to main timeframe)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Calculate 1w RSI for regime filter
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: >2x 24-period MA (2 days of 12h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below KAMA
        uptrend = close[i] > kama_12h_aligned[i]
        downtrend = close[i] < kama_12h_aligned[i]
        
        # Volume confirmation (>2x average)
        vol_confirm = volume[i] > (2.0 * vol_ma_24[i])
        
        # Regime filter: avoid extreme RSI
        rsi_ok = (rsi_1w_aligned[i] > 20) and (rsi_1w_aligned[i] < 80)
        
        # Entry conditions
        long_entry = uptrend and vol_confirm and rsi_ok
        short_entry = downtrend and vol_confirm and rsi_ok
        
        # Exit conditions: trend reversal
        long_exit = not uptrend
        short_exit = not downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_With_Volume_Spike"
timeframe = "12h"
leverage = 1.0