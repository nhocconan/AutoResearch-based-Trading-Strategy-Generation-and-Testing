#!/usr/bin/env python3
# 1D_1W_KAMA_Trend_Stochastic_RSI_Confirmation
# Hypothesis: Use weekly KAMA for trend direction, daily Stochastic RSI for entry timing.
# KAMA adapts to volatility, reducing whipsaw in choppy markets.
# Stochastic RSI identifies overbought/oversold conditions within the trend.
# Works in bull/bear via trend filter. Target: 30-100 total trades over 4 years (7-25/year).

name = "1D_1W_KAMA_Trend_Stochastic_RSI_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Get daily data for Stochastic RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Stochastic RSI: RSI of Stochastic
    stoch_k = 100 * (close_1d - np.minimum.accumulate(low_1d)) / (np.maximum.accumulate(high_1d) - np.minimum.accumulate(low_1d) + 1e-10)
    stoch_k = np.where(np.maximum.accumulate(high_1d) - np.minimum.accumulate(low_1d) == 0, 50, stoch_k)
    
    # RSI of Stochastic
    delta = np.diff(stoch_k, prepend=stoch_k[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    stoch_rsi = 100 * (rsi - np.minimum.accumulate(rsi)) / (np.maximum.accumulate(rsi) - np.minimum.accumulate(rsi) + 1e-10)
    stoch_rsi = np.where(np.maximum.accumulate(rsi) - np.minimum.accumulate(rsi) == 0, 50, stoch_rsi)
    
    # Align to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    stoch_rsi_aligned = align_htf_to_ltf(prices, df_1d, stoch_rsi)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama_aligned[i]) or np.isnan(stoch_rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above weekly KAMA + Stochastic RSI < 20 (oversold) + volume confirmation
            if close[i] > kama_aligned[i] and stoch_rsi_aligned[i] < 20 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly KAMA + Stochastic RSI > 80 (overbought) + volume confirmation
            elif close[i] < kama_aligned[i] and stoch_rsi_aligned[i] > 80 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below weekly KAMA (trend change) OR Stochastic RSI > 80 (overbought)
            if close[i] < kama_aligned[i] or stoch_rsi_aligned[i] > 80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above weekly KAMA (trend change) OR Stochastic RSI < 20 (oversold)
            if close[i] > kama_aligned[i] or stoch_rsi_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals