#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: Use 1d KAMA for trend direction, RSI(14) for momentum, and Choppiness Index for regime filter.
- KAMA(ER=10, FC=30, SC=2) determines primary trend: price > KAMA = bullish, price < KAMA = bearish.
- RSI(14) > 55 confirms bullish momentum, RSI(14) < 45 confirms bearish momentum.
- Choppiness Index(14) < 38.2 confirms trending regime (avoid ranging markets).
- In bullish trend + trending regime: long when RSI > 55.
- In bearish trend + trending regime: short when RSI < 45.
- Exit when trend reverses or regime becomes choppy (CHOP > 61.8).
- Position size: 0.25. Target: 30-100 total trades over 4 years = 7-25/year.
- Works in both bull and bear: KAMA adapts to volatility, regime filter avoids whipsaws in ranging markets.
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA for trend filter
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # This needs correction
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i == 0:
            er[i] = 1.0
        else:
            price_change = np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
            sum_abs_diff = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))) if i >= 1 else np.abs(close_1d[i] - close_1d[0])
            er[i] = price_change / (sum_abs_diff + 1e-10)
    # Smoothing constants
    sc = 2 / (2 + 1)   # Fast SC
    fc = 2 / (30 + 1)  # Slow SC
    ssc = (er * (sc - fc) + fc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + ssc[i] * (close_1d[i] - kama[i-1])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Choppiness Index(14)
    atr_1d = np.zeros_like(close_1d)
    tr = np.maximum(np.maximum(df_1d['high'].values - df_1d['low'].values,
                               np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))),
                      np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1)))
    tr[0] = df_1d['high'].values[0] - df_1d['low'].values[0]
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    max_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = np.where((max_high - min_low) > 0,
                    100 * np.log10(np.sum(atr_1d[-13:]) / (max_high - min_low)) / np.log10(14),
                    50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA, RSI, CHOP
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend and regime
        htf_1d_bullish = close[i] > kama_aligned[i]
        htf_1d_bearish = close[i] < kama_aligned[i]
        trending_regime = chop_aligned[i] < 38.2
        choppy_regime = chop_aligned[i] > 61.8
        
        if position == 0:
            # Entry logic: trade in direction of trend with momentum confirmation
            long_setup = htf_1d_bullish and trending_regime and (rsi_aligned[i] > 55)
            short_setup = htf_1d_bearish and trending_regime and (rsi_aligned[i] < 45)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal or choppy regime
            exit_signal = (not htf_1d_bullish) or choppy_regime
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal or choppy regime
            exit_signal = htf_1d_bullish or choppy_regime
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0