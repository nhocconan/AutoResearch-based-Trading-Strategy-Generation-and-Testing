#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum confirmation and Choppiness Index for regime filtering.
Only trade in KAMA trend direction when RSI is not extreme and market is trending (CHOP < 61.8).
This strategy aims for low trade frequency (7-25/year) by requiring confluence of trend,
momentum, and regime filters, reducing whipsaws in ranging markets and avoiding overextended moves.
Works in bull/bear via trend filter - only long when KAMA trending up, short when trending down.
Chop filter prevents trading in chaotic/ranging markets (CHOP > 61.8 = ranging).
"""

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
    
    # Get 1d data for KAMA, RSI, and Chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA (ER=10, fast=2, slow=30)
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([[np.nan]*10, change])
    volatility = np.concatenate([[np.nan]*10, volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Start after first 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * (max(high_n) - min(low_n)))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero or log of zero
    denominator = 14 * (max_high - min_low)
    chop = np.where(denominator > 0, 100 * np.log10(atr1_sum / denominator) / np.log10(14), 100)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: KAMA direction
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Momentum filter: RSI not extreme (avoid overbought/oversold)
        rsi_not_extreme = (rsi_aligned[i] > 20) & (rsi_aligned[i] < 80)
        
        # Regime filter: only trade when market is trending (CHOP < 61.8 = trending)
        trending_market = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI not extreme, trending market
            if uptrend and rsi_not_extreme and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI not extreme, trending market
            elif downtrend and rsi_not_extreme and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR RSI becomes extreme OR market becomes ranging
            if not uptrend or not rsi_not_extreme or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR RSI becomes extreme OR market becomes ranging
            if not downtrend or not rsi_not_extreme or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0