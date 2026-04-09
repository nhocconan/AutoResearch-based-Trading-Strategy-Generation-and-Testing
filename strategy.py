#!/usr/bin/env python3
# 1d_kama_rsi_chop_v3
# Hypothesis: Daily KAMA trend direction with RSI mean-reversion entries during choppy regimes.
# KAMA adapts to market noise, providing reliable trend direction in both trending and ranging markets.
# RSI extremes (oversold/overbought) within chop regimes offer high-probability mean-reversion trades.
# Chop filter (Choppiness Index > 61.8) ensures we only trade in ranging markets where mean reversion works.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for regime filter (chop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) - trend direction
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:14])  # 14-period volatility
    for i in range(14, len(close)):
        volatility = volatility - np.abs(close[i-14] - close[i-14]) + np.abs(close[i] - close[i-1])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) from weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    atr_1w = pd.Series(high_1w - low_1w).rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero and log10 of zero
    chop_denom = np.log10(atr_1w) * np.log10(14)
    chop_denom = np.where(chop_denom <= 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime: only trade when market is ranging (chop > 61.8)
        chop_regime = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete) or trend changes
            if rsi[i] >= 50 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete) or trend changes
            if rsi[i] <= 50 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if chop_regime:
                # Long entry: RSI oversold (<30) and price above KAMA (uptrend bias)
                if rsi[i] < 30 and close[i] > kama[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI overbought (>70) and price below KAMA (downtrend bias)
                elif rsi[i] > 70 and close[i] < kama[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals