#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI(2) extremes + 1w ADX regime filter
# KAMA adapts to market efficiency, providing smooth trend direction with less whipsaw.
# RSI(2) identifies short-term overextensions for mean-reversion entries in the direction of KAMA trend.
# 1w ADX > 25 ensures we only trade in trending regimes (avoiding chop).
# Designed for 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Works in bull markets (buying pullbacks in uptrend) and bear markets (selling rallies in downtrend)
# by only taking mean-reversion entries aligned with the weekly trend.

name = "1d_KAMA_RSI2_1wADX25_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 1d KAMA for trend direction
    # KAMA: Efficiency Ratio (ER) smoothed with fast/slow SC
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[-10:])  # 10-period volatility
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 1w ADX for regime filter (trending vs ranging)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1w, prepend=high_1w[0]))
    tr2 = np.abs(np.diff(low_1w, prepend=low_1w[0]))
    tr3 = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate RSI(2) for mean-reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30  # need enough data for KAMA, RSI(2), and aligned ADX
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(kama[i]) or np.isnan(rsi_2[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending regimes (ADX > 25 on weekly)
        if adx_1w_aligned[i] <= 25:
            # In ranging markets, flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price < KAMA (pullback in uptrend) AND RSI(2) < 10 (extreme oversold)
            if close[i] < kama[i] and rsi_2[i] < 10:
                signals[i] = 0.25
                position = 1
            # Short entry: price > KAMA (rally in downtrend) AND RSI(2) > 90 (extreme overbought)
            elif close[i] > kama[i] and rsi_2[i] > 90:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price >= KAMA (trend resumption) OR RSI(2) > 50 (mean reversion complete)
            if close[i] >= kama[i] or rsi_2[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price <= KAMA (trend resumption) OR RSI(2) < 50 (mean reversion complete)
            if close[i] <= kama[i] or rsi_2[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals