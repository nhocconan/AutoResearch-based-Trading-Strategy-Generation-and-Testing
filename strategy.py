#!/usr/bin/env python3
# 12h_KAMA_RSI_Trend_Filter
# Hypothesis: On 12h timeframe, KAMA captures trend direction while RSI filters overextended entries.
# In trending markets (KAMA slope > 0), go long on RSI pullbacks from overbought (RSI < 60).
# In downtrends (KAMA slope < 0), go short on RSI bounces from oversold (RSI > 40).
# Uses 1w EMA50 as higher timeframe trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~15-25/year) to minimize fee drag and work in both bull/bear markets.

name = "12h_KAMA_RSI_Trend_Filter"
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
    
    # Get weekly data for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    # Weekly EMA50 for trend filter
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily data
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    
    # Proper ER calculation
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if i >= 1:
            direction = np.abs(close_1d[i] - close_1d[i-10]) if i >= 10 else np.abs(close_1d[i] - close_1d[0])
            volatility = np.sum(np.abs(np.diff(close_1d[max(0,i-9):i+1])))
            if volatility > 0:
                er[i] = direction / volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on daily data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly EMA50 to 12h
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align daily KAMA and RSI to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: only trade in direction of weekly trend
        weekly_uptrend = close_1w[-1] > ema50_1w_aligned[i] if len(close_1w) > 0 else False
        weekly_downtrend = close_1w[-1] < ema50_1w_aligned[i] if len(close_1w) > 0 else False
        
        # Daily signals
        price_above_kama = close[i] > kama_aligned[i]
        price_below_kama = close[i] < kama_aligned[i]
        rsi_not_overbought = rsi_aligned[i] < 60  # Avoid buying too overextended
        rsi_not_oversold = rsi_aligned[i] > 40    # Avoid selling too overextended
        
        if position == 0:
            # Enter long: weekly uptrend + price above KAMA + RSI not overbought
            if weekly_uptrend and price_above_kama and rsi_not_overbought:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly downtrend + price below KAMA + RSI not oversold
            elif weekly_downtrend and price_below_kama and rsi_not_oversold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly downtrend or price below KAMA
            if not weekly_uptrend or price_below_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly uptrend or price above KAMA
            if not weekly_downtrend or price_above_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals