#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA (Kaufman Adaptive Moving Average) with RSI(14) filter and chop regime filter.
KAMA adapts to market noise - slows in ranging markets, speeds in trending markets.
Combined with RSI for momentum confirmation and Choppiness Index to avoid false signals in high-chop environments.
Designed for low frequency (~10-25 trades/year) to minimize fee drag, works in bull/bear via adaptive trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly RSI(14) for regime filter - avoids choppy weekly regimes
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Load 1d data ONCE before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA(10, 2, 30) - Kaufman Adaptive Moving Average
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, n=10, prepend=close_1d[:10]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)
    # Fix volatility calculation - rolling sum of absolute changes
    volatility = pd.Series(np.abs(np.diff(close_1d, prepend=close_1d[0]))).rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        rsi_1w_val = rsi_1w_aligned[i]
        
        # Weekly regime filter: avoid extreme RSI levels that indicate exhaustion
        weekly_regime_ok = (rsi_1w_val > 20) and (rsi_1w_val < 80)
        
        if position == 0:
            # Enter long: price > KAMA, RSI > 50 (bullish momentum), weekly not overbought
            if (price_close > kama_val and 
                rsi_val > 50 and 
                weekly_regime_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA, RSI < 50 (bearish momentum), weekly not oversold
            elif (price_close < kama_val and 
                  rsi_val < 50 and 
                  weekly_regime_ok):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses KAMA in opposite direction or RSI extreme
            if position == 1 and (price_close < kama_val or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > kama_val or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_WeeklyFilter"
timeframe = "1d"
leverage = 1.0