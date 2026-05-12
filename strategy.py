#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
Hypothesis: On daily timeframe, use Kaufman's Adaptive Moving Average (KAMA) to determine trend direction.
Enter long when KAMA slopes upward and RSI(14) > 50, with chop filter avoiding ranging markets.
Enter short when KAMA slopes downward and RSI(14) < 50.
Exit when trend reverses or chop increases.
Uses weekly trend filter to align with higher timeframe momentum.
Designed for low trade frequency (<25/year) to minimize fee drag, works in bull via trend continuation
and in bear via counter-trend reversals at overextended RSI levels.
"""

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio and Smoothing Constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation properly
    volatility_series = pd.Series(volatility)
    volatility_rolling = volatility_series.rolling(window=10, min_periods=10).sum().values
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility_rolling != 0
    er[mask] = change[mask] / volatility_rolling[mask]
    er[~mask] = 0
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14)
    atr = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First true range
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros_like(close)
    mask = (hh - ll) != 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / (hh[mask] - ll[mask])) / np.log10(14)
    chop[~mask] = 50  # Neutral when no range
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA to daily
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Align KAMA, RSI, Chop to daily (already same timeframe, but ensure alignment)
    # For same timeframe, alignment is trivial but we keep for consistency
    kama_aligned = kama  # Already aligned to prices
    rsi_aligned = rsi
    chop_aligned = chop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        kama_val = kama_aligned[i]
        kama_prev = kama_aligned[i-1]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        weekly_trend = ema20_1w_aligned[i]
        
        # KAMA slope: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # LONG: KAMA rising, RSI > 50, chop < 61.8 (trending), price > weekly EMA20
            if (kama_rising and rsi_val > 50 and chop_val < 61.8 and close[i] > weekly_trend):
                signals[i] = 0.25
                position = 1
            # SHORT: KAMA falling, RSI < 50, chop < 61.8, price < weekly EMA20
            elif (kama_falling and rsi_val < 50 and chop_val < 61.8 and close[i] < weekly_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: KAMA falling OR RSI < 40 OR chop > 61.8 (ranging) OR price < weekly EMA20
            if (kama_falling or rsi_val < 40 or chop_val > 61.8 or close[i] < weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: KAMA rising OR RSI > 60 OR chop > 61.8 OR price > weekly EMA20
            if (kama_rising or rsi_val > 60 or chop_val > 61.8 or close[i] > weekly_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals