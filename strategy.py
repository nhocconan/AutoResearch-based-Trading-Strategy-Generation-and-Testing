#!/usr/bin/env python3
# 1d_kama_rsi_chop_v2
# Hypothesis: 1d strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
# RSI(14) for momentum confirmation, and Choppiness Index for regime filter.
# Trades only in ranging markets (CHOP > 50) to avoid whipsaws in strong trends.
# KAMA filters noise and adapts to volatility. RSI confirms momentum alignment.
# Discrete sizing (0.0, ±0.25) to minimize fee churn. Target: 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v2"
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
    
    # 1w HTF data for higher timeframe context (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average) on close
    close_s = pd.Series(close)
    change = np.abs(close_s.diff(10).values)  # 10-period net change
    volatility = np.abs(close_s.diff(1).rolling(window=10, min_periods=10).sum().values)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.06645 - 0.0645) + 0.0645) ** 2  # smoothing constants
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14-period) from 1d data
    atr = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(atr) * np.log10(14)
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    
    # Calculate 1w EMA(50) for trend filter (aligned to 1d)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (chop > 50)
        chop_regime = chop[i] > 50
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or RSI < 40 (momentum loss)
            if close[i] < kama[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or RSI > 60 (momentum loss)
            if close[i] > kama[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if chop_regime:
                # Long entry: price above KAMA, RSI > 50 (bullish momentum), and above 1w EMA50 (uptrend bias)
                if close[i] > kama[i] and rsi[i] > 50 and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price below KAMA, RSI < 50 (bearish momentum), and below 1w EMA50 (downtrend bias)
                elif close[i] < kama[i] and rsi[i] < 50 and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals