#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1D KAMA + RSI + Chop Filter
# Hypothesis: Kaufman's Adaptive Moving Average (KAMA) adapts to market noise,
# providing a dynamic trend filter. Combined with RSI for momentum and Choppiness
# Index to avoid ranging markets, this strategy aims to capture strong trends
# while avoiding whipsaw in both bull and bear markets. Daily timeframe reduces
# trade frequency to minimize fee drag.
name = "1d_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # KAMA (Kaufman's Adaptive Moving Average)
    # Efficiency Ratio (ER) = |change| / volatility
    change = np.abs(np.diff(close, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # 10-period volatility
    # Handle the array shapes properly
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # RSI (Relative Strength Index)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index
    # True Range
    tr1 = high - low
    tr2 = np.abs(np.concatenate([[high[0]], high[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(np.concatenate([[low[0]], low[:-1]]) - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Sum of True Range over period
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Choppiness Index
    chop = np.where(atr_sum != 0, 100 * np.log10((hh - ll) / atr_sum) / np.log10(14), 50)
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    # Weekly EMA for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(weekly_ema_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price below KAMA or RSI overbought or choppy market
            if close[i] < kama[i] or rsi[i] > 70 or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price above KAMA or RSI oversold or choppy market
            if close[i] > kama[i] or rsi[i] < 30 or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require trending market (low chop) and alignment
            if chop[i] < 38.2:  # Trending market
                # Long: price above KAMA + RSI bullish + weekly uptrend
                if close[i] > kama[i] and rsi[i] > 50 and close[i] > weekly_ema_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price below KAMA + RSI bearish + weekly downtrend
                elif close[i] < kama[i] and rsi[i] < 50 and close[i] < weekly_ema_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals