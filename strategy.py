#!/usr/bin/env python3
"""
1d_kama_rsi_chop_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI for mean reversion signals, and Choppiness Index to filter ranging vs trending markets.
In ranging markets (CHOP > 61.8), fade extreme RSI with KAMA trend filter.
In trending markets (CHOP < 38.2), follow KAMA direction with pullbacks.
This adapts to both bull and bear regimes via dynamic trend and market state filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Weekly data for trend filter and regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly close for KAMA
    close_1w = df_1w['close'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio and KAMA
    change = np.abs(np.diff(close_1w, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)  # 10-period volatility
    # Avoid division by zero
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama = np.full_like(close_1w, np.nan, dtype=float)
    kama[9] = close_1w[9]  # Start after 10 periods
    for i in range(10, len(close_1w)):
        kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Weekly RSI (14)
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Weekly Choppiness Index (14)
    atr = np.zeros_like(close_1w)
    tr1 = np.abs(np.diff(close_1w))
    tr2 = np.abs(np.diff(df_1w['high'].values))
    tr3 = np.abs(np.diff(df_1w['low'].values))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Prepend NaN for first 14 values
    atr = np.concatenate([np.full(14, np.nan), atr])
    
    max_high = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop = np.where((max_high - min_low) > 0, 
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 
                    50)
    # Prepend NaN for first 14 values
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    # Align weekly indicators to daily timeframe
    kama_1d = align_htf_to_ltf(prices, df_1w, kama)
    rsi_1d = align_htf_to_ltf(prices, df_1w, rsi)
    chop_1d = align_htf_to_ltf(prices, df_1w, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Warmup for indicators
        # Skip if required data not available
        if (np.isnan(kama_1d[i]) or np.isnan(rsi_1d[i]) or np.isnan(chop_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) OR chop > 61.8 and price < KAMA (trend fail in range)
            if rsi_1d[i] > 70 or (chop_1d[i] > 61.8 and close[i] < kama_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) OR chop > 61.8 and price > KAMA (trend fail in range)
            if rsi_1d[i] < 30 or (chop_1d[i] > 61.8 and close[i] > kama_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Chop > 61.8 = ranging market: mean reversion at RSI extremes
            if chop_1d[i] > 61.8:
                if rsi_1d[i] < 30 and close[i] > kama_1d[i]:  # Oversold + above trend
                    position = 1
                    signals[i] = 0.25
                elif rsi_1d[i] > 70 and close[i] < kama_1d[i]:  # Overbought + below trend
                    position = -1
                    signals[i] = -0.25
            # Chop < 38.2 = trending market: follow KAMA direction
            elif chop_1d[i] < 38.2:
                if close[i] > kama_1d[i]:  # Uptrend
                    position = 1
                    signals[i] = 0.25
                elif close[i] < kama_1d[i]:  # Downtrend
                    position = -1
                    signals[i] = -0.25
    
    return signals