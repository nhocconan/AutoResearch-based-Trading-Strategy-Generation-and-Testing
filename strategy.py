#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: Daily KAMA direction + RSI(14) extremes + Choppiness Index regime filter.
Long when KAMA trending up, RSI < 30 (oversold) and CHOP > 61.8 (choppy market favors mean reversion).
Short when KAMA trending down, RSI > 70 (overbought) and CHOP > 61.8.
Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Designed for low trade frequency (target: 7-25/year) with discrete sizing (0.25) to minimize fee drag.
Works in both bull and bear markets by combining trend (KAMA/1wEMA) with mean reversion (RSI extremes) in choppy regimes.
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
    volume = prices['volume'].values
    
    # Get 1d data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicators ===
    # KAMA (Kaufman Adaptive Moving Average) - ER=10, Fast=2, Slow=30
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Choppiness Index (CHOP) - 14 period
    atr_raw = np.maximum(high - low, np.absolute(high - np.roll(close_1d, 1)), np.absolute(low - np.roll(close_1d, 1)))
    atr_raw[0] = high[0] - low[0]
    atr_sum = pd.Series(atr_raw).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 1w Trend Filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for KAMA/RSI/CHOP)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: KAMA up, RSI oversold, choppy market
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] > 61.8 and
                uptrend_1w[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought, choppy market
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] > 61.8 and
                  downtrend_1w[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI > 50 (mean reversion complete) OR 1w trend changes
            if (close[i] < kama_aligned[i] or 
                rsi_aligned[i] > 50 or 
                not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI < 50 (mean reversion complete) OR 1w trend changes
            if (close[i] > kama_aligned[i] or 
                rsi_aligned[i] < 50 or 
                not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0