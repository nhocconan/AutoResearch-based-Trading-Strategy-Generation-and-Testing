#!/usr/bin/env python3
"""
1d RSI + Bollinger Bands Reversal
Hypothesis: In daily timeframe, RSI extremes combined with Bollinger Band touches
identify exhaustion points. Long when RSI < 30 and price touches lower BB,
short when RSI > 70 and price touches upper BB. Works in both bull and bear
markets by capturing mean reversals from overextended conditions. Uses 1w trend
filter to avoid counter-trend trades. Target: 10-20 trades/year to minimize fee drag.
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
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands (20, 2)
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # 1-week trend filter (SMA 50)
    df_1w = get_htf_data(prices, '1w')
    sma_50_1w = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(sma_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        sma_val = sma[i]
        upper_val = upper[i]
        lower_val = lower[i]
        trend_1w = sma_50_1w_aligned[i]
        
        if position == 0:
            # Long: oversold RSI + touches lower BB + above weekly trend
            if rsi_val < 30 and price <= lower_val and price > trend_1w:
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI + touches upper BB + below weekly trend
            elif rsi_val > 70 and price >= upper_val and price < trend_1w:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if RSI returns to neutral or price crosses above SMA
            if rsi_val > 50 or price > sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if RSI returns to neutral or price crosses below SMA
            if rsi_val < 50 or price < sma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_RSI_Bollinger_Reversal"
timeframe = "1d"
leverage = 1.0