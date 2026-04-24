#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend + RSI mean reversion + chop regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for regime filter (chop > 61.8 = range, chop < 38.2 = trend).
- KAMA(10) for adaptive trend direction (bullish if close > KAMA, bearish if close < KAMA).
- RSI(14) for mean reversion entries (long when RSI < 30, short when RSI > 70).
- Only trade in choppy regimes (CHOP > 61.8) to avoid whipsaws in strong trends.
- Exit on RSI reversal (long exit when RSI > 50, short exit when RSI < 50).
- Signal size: 0.25 discrete to balance profit and drawdown.
Designed for BTC/ETH: mean reversion works in ranging/choppy markets, trend filter avoids counter-trend trades.
Proven pattern from DB: KAMA + RSI + chop filter shows strong test performance on SOL (Sharpe=1.31).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA(10) for trend filter
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = close_1d[i]
    
    # Calculate 1d RSI(14) for mean reversion
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Get 1w data for chop regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w Chopiness Index(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    atr_1w = np.zeros_like(close_1w)
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w[1:] = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of ATR over 14 periods
    sum_atr_1w = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
    # Max high - min low over 14 periods
    max_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_1w = max_high_1w - min_low_1w
    # Chopiness Index
    chop = np.where(range_1w != 0, 100 * np.log10(sum_atr_1w / range_1w) / np.log10(14), 50)
    
    # Align HTF indicators to 1d
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 30, 14, 14)  # Need enough bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Check for entry signals in choppy regime only
            in_chop = chop_aligned[i] > 61.8
            
            # Long: Close below KAMA (trend filter) AND RSI < 30 (oversold) AND in chop
            if curr_close < kama_aligned[i] and rsi_aligned[i] < 30 and in_chop:
                signals[i] = 0.25
                position = 1
            # Short: Close above KAMA (trend filter) AND RSI > 70 (overbought) AND in chop
            elif curr_close > kama_aligned[i] and rsi_aligned[i] > 70 and in_chop:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when RSI > 50 (mean reversion complete)
            if rsi_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when RSI < 50 (mean reversion complete)
            if rsi_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_MeanReversion_v1"
timeframe = "1d"
leverage = 1.0