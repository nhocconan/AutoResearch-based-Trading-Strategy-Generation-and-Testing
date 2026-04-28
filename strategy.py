#!/usr/bin/env python3
"""
12h_KAMA_Trend_RSI_Chop
Hypothesis: On 12-hour timeframe, use Kaufman Adaptive Moving Average (KAMA) to capture trend direction with minimal whipsaw. Enter long when price crosses above KAMA with RSI > 50 and choppy market filter (Chop > 61.8), short when price crosses below KAMA with RSI < 50 and Chop > 61.8. Exit on opposite cross. Uses 1-day trend filter (price > 1d EMA50 for long, price < 1d EMA50 for short) to align with higher timeframe trend. Designed for low trade frequency (~20-40/year) to minimize fee decay in both bull and bear markets. KAMA adapts to market noise, reducing false signals in ranging conditions.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    # Parameters: ER length=10, Fast SC=2, Slow SC=30
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER and volatility calculation
    er = np.zeros(n)
    volatility = np.zeros(n)
    for i in range(10, n):
        change_val = np.abs(close[i] - close[i-10])
        volatility_val = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if volatility_val > 0:
            er[i] = change_val / volatility_val
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # (ER * (fast - slow) + slow)^2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14-period) - measures ranging vs trending
    # High values (>61.8) indicate ranging/choppy market (good for mean reversion)
    # Low values (<38.2) indicate trending market
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_min_range = pd.Series(np.maximum(high, np.roll(close, 1)) - np.minimum(low, np.roll(close, 1))).rolling(window=14, min_periods=14).max().values - \
                    pd.Series(np.minimum(low, np.roll(close, 1)) - np.maximum(high, np.roll(close, 1))).rolling(window=14, min_periods=14).min().values
    # Fix the range calculation
    true_high = np.maximum(high, np.roll(close, 1))
    true_low = np.minimum(low, np.roll(close, 1))
    max_min_range = pd.Series(true_high - true_low).rolling(window=14, min_periods=14).max().values - \
                    pd.Series(true_high - true_low).rolling(window=14, min_periods=14).min().values
    
    chop = np.zeros(n)
    for i in range(n):
        if atr_sum[i] > 0 and max_min_range[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / max_min_range[i]) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Trend filter: bullish when price > 1d EMA50, bearish when price < 1d EMA50
    bullish_trend = close > ema_50_1d_aligned
    bearish_trend = close < ema_50_1d_aligned
    
    # Chop filter: only trade in choppy/ranging markets (Chop > 61.8)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] > kama[i]) and (rsi[i] > 50) and chop_filter[i] and bullish_trend[i]
        short_entry = (close[i] < kama[i]) and (rsi[i] < 50) and chop_filter[i] and bearish_trend[i]
        
        # Exit on opposite cross
        long_exit = close[i] < kama[i]
        short_exit = close[i] > kama[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_KAMA_Trend_RSI_Chop"
timeframe = "12h"
leverage = 1.0