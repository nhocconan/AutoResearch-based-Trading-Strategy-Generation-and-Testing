#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: Daily KAMA + RSI + Chop Filter
# Hypothesis: KAMA adapts to market noise, providing reliable trend direction in both bull and bear markets.
# RSI filters for overbought/oversold conditions, while Choppiness Index filters for trending markets only.
# Works in bull markets: KAMA up + RSI > 50 + chop < 61.8 = long
# Works in bear markets: KAMA down + RSI < 50 + chop < 61.8 = short
# Weekly trend filter ensures we only trade with the higher timeframe trend.
# Target: 15-25 trades/year (60-100 over 4 years).

name = "daily_kama_rsi_chop_weekly_trend_v1"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close_10|
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # sum of absolute changes
    # Fix volatility calculation: need rolling sum of absolute changes
    price_change = np.abs(np.diff(close, n=1))
    volatility = np.convolve(price_change, np.ones(10), mode='same')
    volatility[:10] = volatility[10]  # pad beginning
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI (14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Choppiness Index (14)
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr[1:] = np.truediv(np.convolve(tr, np.ones(14), mode='same'), 14)
    atr[:14] = atr[14]  # pad beginning
    
    # True range for chop calculation
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Sum of true ranges over 14 periods
    tr_sum = np.convolve(tr, np.ones(14), mode='same')
    tr_sum[:13] = tr_sum[13]  # pad beginning
    
    # Max high - min low over 14 periods
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 13)
        end_idx = i + 1
        max_high[i] = np.max(high[start_idx:end_idx])
        min_low[i] = np.min(low[start_idx:end_idx])
    
    # Avoid division by zero
    range_14 = max_high - min_low
    chop = np.where(range_14 != 0, 100 * np.log10(tr_sum / range_14) / np.log10(14), 50)
    
    # Get weekly trend: EMA crossover
    weekly_close = df_weekly['close'].values
    ema_fast = pd.Series(weekly_close).ewm(span=9, adjust=False).mean().values
    ema_slow = pd.Series(weekly_close).ewm(span=21, adjust=False).mean().values
    
    # Align weekly EMA to daily
    ema_fast_aligned = align_htf_to_ltf(prices, df_weekly, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_weekly, ema_slow)
    weekly_uptrend = ema_fast_aligned > ema_slow_aligned
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA turns down OR RSI < 40 OR chop > 61.8 (ranging) OR weekly trend turns down
            if (close[i] < kama[i] or rsi[i] < 40 or chop[i] > 61.8 or not weekly_uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: KAMA turns up OR RSI > 60 OR chop > 61.8 (ranging) OR weekly trend turns up
            if (close[i] > kama[i] or rsi[i] > 60 or chop[i] > 61.8 or weekly_uptrend[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: KAMA up AND RSI > 50 AND chop < 61.8 (trending) AND weekly uptrend
            if (close[i] > kama[i] and rsi[i] > 50 and chop[i] < 61.8 and weekly_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Short: KAMA down AND RSI < 50 AND chop < 61.8 (trending) AND weekly downtrend
            elif (close[i] < kama[i] and rsi[i] < 50 and chop[i] < 61.8 and not weekly_uptrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals