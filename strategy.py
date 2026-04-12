#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_rsi_vs_ma_mean_reversion
# Uses RSI(14) mean reversion with daily moving average filter.
# Long when RSI < 30 and price > daily EMA(50) (oversold in uptrend).
# Short when RSI > 70 and price < daily EMA(50) (overbought in downtrend).
# Works in both bull and bear markets by combining momentum exhaustion with trend filter.
# Target: 20-40 trades/year per symbol.

name = "6h_1d_rsi_vs_ma_mean_reversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate RSI(14) on 6h closes
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if EMA not ready
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Long: RSI oversold and price above daily EMA
        if rsi[i] < 30 and close[i] > ema_50_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: RSI overbought and price below daily EMA
        elif rsi[i] > 70 and close[i] < ema_50_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI returns to neutral zone
        elif 40 <= rsi[i] <= 60 and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals