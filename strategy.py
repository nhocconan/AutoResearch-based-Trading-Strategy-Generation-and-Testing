#!/usr/bin/env python3
"""
1d_1w_kama_rsi_v1
Hypothesis: 1-day KAMA trend with RSI mean reversion on weekly timeframe, filtered by 1-week volatility regime.
- KAMA adapts to market noise, effective in both trending and ranging markets.
- Weekly RSI > 70 or < 30 indicates overextension on higher timeframe, signaling mean reversion.
- Weekly ATR ratio (current/average) filters for high volatility environments where mean reversion works best.
- Position size: 0.25 for mean reversion trades.
Target: 15-25 trades/year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_len=10, fast_len=2, slow_len=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).rolling(window=er_len, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for RSI and ATR
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly RSI(14)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Weekly ATR(14) and its average for volatility regime
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / atr_ma_1w
    
    # Daily KAMA(10,2,30)
    kama = calculate_kama(close, er_len=10, fast_len=2, slow_len=30)
    
    # Align weekly indicators to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    signals = np.zeros(n)
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if weekly data not available
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i]):
            signals[i] = 0.0
            continue
        
        # High volatility regime (ATR ratio > 1.2) for mean reversion
        if atr_ratio_aligned[i] > 1.2:
            # Mean reversion signals from weekly RSI extremes
            if rsi_1w_aligned[i] < 30 and close[i] > kama[i]:
                # Oversold and price above KAMA -> long
                signals[i] = 0.25
            elif rsi_1w_aligned[i] > 70 and close[i] < kama[i]:
                # Overbought and price below KAMA -> short
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:
            # Low volatility: follow KAMA trend
            if close[i] > kama[i]:
                signals[i] = 0.25
            elif close[i] < kama[i]:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals