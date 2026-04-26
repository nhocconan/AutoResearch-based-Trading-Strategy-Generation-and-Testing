#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction,
RSI(14) for momentum confirmation, and Choppiness Index(14) for regime filter.
Enter long when KAMA slopes up, RSI > 50, and market is trending (CHOP < 38.2).
Enter short when KAMA slopes down, RSI < 50, and market is trending (CHOP < 38.2).
Exit when trend reverses or market becomes choppy (CHOP > 61.8).
Uses 1-week EMA200 as higher timeframe trend filter to avoid counter-trend trades.
Position size: 0.25 (discrete to minimize fee churn). Target: 30-100 trades over 4 years (7-25/year).
Designed to work in both bull and bear markets by requiring trend alignment across timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Calculate KAMA (trend indicator) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close - close[1]| over 10 periods
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    fastest = 2/(2+1)   # for EMA 2
    slowest = 2/(30+1)  # for EMA 30
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === Calculate RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain, dtype=float), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # First 14 values are NaN due to min_periods
    
    # === Calculate Choppiness Index(14) ===
    # True Range
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Max/min close over 14 periods
    max_close = pd.Series(close).rolling(window=14, min_periods=14).max().values
    min_close = pd.Series(close).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum(atr)/ (max_close - min_close)) / log10(14)
    range_ = max_close - min_close
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = np.divide(
        100 * np.log10(sum_atr / range_),
        np.log10(14),
        out=np.full_like(sum_atr, np.nan),
        where=(range_ != 0) & (~np.isnan(range_)) & (~np.isnan(sum_atr))
    )
    
    # === Load 1-week EMA200 for HTF trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 30 for KAMA, 14 for RSI/Chop, 200 for 1w EMA)
    start_idx = max(30, 14, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # KAMA direction: slope over 2 periods
        kama_rising = kama[i] > kama[i-2]
        kama_falling = kama[i] < kama[i-2]
        
        # Regime filter: trending market (CHOP < 38.2)
        trending = chop[i] < 38.2
        choppy = chop[i] > 61.8
        
        # Higher timeframe trend filter: price vs 1w EMA200
        uptrend_htf = close[i] > ema_200_1w_aligned[i]
        downtrend_htf = close[i] < ema_200_1w_aligned[i]
        
        # Long conditions: KAMA up, RSI > 50, trending, and HTF uptrend
        long_condition = kama_rising and (rsi[i] > 50) and trending and uptrend_htf
        # Short conditions: KAMA down, RSI < 50, trending, and HTF downtrend
        short_condition = kama_falling and (rsi[i] < 50) and trending and downtrend_htf
        
        # Exit conditions: trend reversal or choppy market
        exit_long = not kama_rising or choppy or not uptrend_htf
        exit_short = not kama_falling or choppy or not downtrend_htf
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0