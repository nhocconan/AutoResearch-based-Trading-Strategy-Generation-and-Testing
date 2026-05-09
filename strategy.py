#!/usr/bin/env python3
name = "12H_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # Get daily data for weekly trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA (5-period on daily = ~25 days) for trend filter
    ema5_1d = pd.Series(close_1d).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Align daily EMA5 to 12h timeframe
    ema5_1d_aligned = align_htf_to_ltf(prices, df_1d, ema5_1d)
    
    # Calculate KAMA on 12h timeframe
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = change[10:] / np.maximum(volatility[10:], 1e-10)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14) for filter
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(30, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema5_1d_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above daily EMA5
        uptrend = close[i] > ema5_1d_aligned[i]
        # Downtrend: price below daily EMA5
        downtrend = close[i] < ema5_1d_aligned[i]
        # KAMA direction: price above/below KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        # RSI filter: avoid extremes
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        if position == 0:
            # Enter long: Uptrend + price above KAMA + RSI not overbought
            if uptrend and price_above_kama and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price below KAMA + RSI not oversold
            elif downtrend and price_below_kama and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price crosses below KAMA
            if not uptrend or not price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price crosses above KAMA
            if not downtrend or not price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals