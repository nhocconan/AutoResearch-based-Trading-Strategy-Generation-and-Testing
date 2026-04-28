#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter
Hypothesis: 1-day Kaufman Adaptive Moving Average (KAMA) identifies trend direction,
RSI(14) filters overextended entries, and Choppiness Index (CHOP) avoids ranging markets.
KAMA adapts to market noise, reducing whipsaws in chop while capturing trends.
RSI extremes with trend filter prevent buying strength/selling weakness.
CHOP > 61.8 = range (avoid), CHOP < 38.2 = trend (trade). Targets 15-25 trades/year.
Works in bull/bear: trend following in trends, avoids false signals in ranges.
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
    
    # Get weekly data for trend filter (more stable than daily)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    kama_1w = calculate_kama(close_1w, 10, 2, 30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Daily RSI(14) for entry filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily Choppiness Index (14)
    atr_14 = calculate_atr(high, low, close, 14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly KAMA
        uptrend = close[i] > kama_1w_aligned[i]
        downtrend = close[i] < kama_1w_aligned[i]
        
        # RSI filter: avoid extremes, look for pullbacks in trend
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_neutral = (rsi[i] >= 40) & (rsi[i] <= 60)
        
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending = chop[i] < 38.2
        
        # Entry logic: pullback in trend with RSI not extreme
        long_entry = trending and uptrend and rsi_oversold
        short_entry = trending and downtrend and rsi_overbought
        
        # Exit logic: trend change or opposite RSI extreme
        long_exit = (not uptrend) or rsi_overbought
        short_exit = (not downtrend) or rsi_oversold
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

def calculate_kama(close, fast_sc, slow_sc, lookback):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, lookback, prepend=close[:lookback]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if lookback == 1 else \
                 pd.Series(np.abs(np.diff(close))).rolling(lookback, min_periods=1).sum().values
    er = np.where(volatility != 0, change / volatility, 0)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def calculate_atr(high, low, close, period):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

name = "1d_KAMA_RSI_Chop_Filter"
timeframe = "1d"
leverage = 1.0