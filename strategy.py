#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week KAMA trend filter with 1-day RSI mean reversion and volume confirmation.
In uptrend (price > weekly KAMA), buy when RSI < 30 (oversold); in downtrend (price < weekly KAMA), sell when RSI > 70 (overbought).
Uses weekly KAMA for trend (avoids whipsaw) and RSI for mean reversion entries. Volume > 1.5x 20-period average confirms momentum.
Target: ~15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
Works in bull via buying dips in uptrend and in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop for KAMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly KAMA (ER=10, slow=30, fast=2)
    close_1w = df_1w['close'].values
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1w)
    for i in range(10, len(close_1w)):
        direction = np.abs(close_1w[i] - close_1w[i-10])
        volatility = np.sum(np.abs(np.diff(close_1w[i-9:i+1])))
        er[i] = direction / volatility if volatility > 0 else 0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smooth constant
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Align weekly KAMA to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # Daily RSI(14)
    delta = np.diff(prices['close'].values, prepend=prices['close'].values[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation (>1.5x 20-day average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5
        
        if position == 0:
            # Enter long: price > weekly KAMA (uptrend) + RSI < 30 (oversold) + volume confirmation
            if (price_close > kama_val and 
                rsi_val < 30 and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price < weekly KAMA (downtrend) + RSI > 70 (overbought) + volume confirmation
            elif (price_close < kama_val and 
                  rsi_val > 70 and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI crosses 50 in opposite direction (mean reversion complete)
            if position == 1 and rsi_val > 50:
                signals[i] = 0.0
                position = 0
            elif position == -1 and rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0