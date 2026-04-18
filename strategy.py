#!/usr/bin/env python3
"""
1d KAMA + RSI + Chop Filter
Hypothesis: KAMA adapts to market noise, providing a smoothed trend line.
In trending markets, price stays above/below KAMA; in ranging markets, it crosses frequently.
Combined with RSI extremes and Choppiness Index regime filter, this strategy:
- Goes long when price is above KAMA, RSI > 50, and market is trending (CHOP < 38.2)
- Goes short when price is below KAMA, RSI < 50, and market is trending (CHOP < 38.2)
- Avoids ranging markets where whipsaws occur.
Designed for low trade frequency and robustness in both bull and bear markets.
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
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0) if hasattr(np, 'sum') else None
    # Manual calculation for volatility sum over 10 periods
    volatility_sum = np.zeros_like(close)
    for i in range(n):
        if i < 10:
            volatility_sum[i] = np.nan
        else:
            volatility_sum[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
    er = np.where(volatility_sum != 0, change / volatility_sum, 0)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (CHOP) over 14 periods
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # ATR14
    atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    # Sum of TR over 14 periods
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop calculation
    chop = 100 * np.log10(sum_tr14 / (atr14 * 14)) / np.log10(14)
    
    # Get 1-week data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1-week EMA50 for higher timeframe trend
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1w EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        # Chop filter: only trade in trending markets (CHOP < 38.2)
        trending_market = chop[i] < 38.2
        
        if position == 0:
            # Enter long: price above KAMA, RSI > 50, uptrend on 1w, trending market
            if close[i] > kama[i] and rsi[i] > 50 and uptrend_1w and trending_market:
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA, RSI < 50, downtrend on 1w, trending market
            elif close[i] < kama[i] and rsi[i] < 50 and downtrend_1w and trending_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA OR RSI < 40 OR market becomes ranging
            if close[i] < kama[i] or rsi[i] < 40 or chop[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA OR RSI > 60 OR market becomes ranging
            if close[i] > kama[i] or rsi[i] > 60 or chop[i] >= 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0