#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_With_Chop_Filter
Hypothesis: KAMA adapts to market efficiency, providing a robust trend filter. Combined with RSI momentum
and a chop filter to avoid ranging markets, this strategy captures trending moves in both bull and bear
markets while avoiding false signals during low volatility periods. Uses weekly trend for higher timeframe
confirmation to reduce false signals. Target: 10-25 trades/year (40-100 total over 4 years).
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
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros(n)
    er[er_period-1:] = change[er_period-1:] / volatility[er_period-1:]
    er[er_period-1:] = np.where(volatility[er_period-1:] == 0, 0, er[er_period-1:])
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    sc[er_period-1:] = sc[er_period-1:]
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[er_period-1] = close[er_period-1]
    for i in range(er_period, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[rsi_period] = np.mean(gain[1:rsi_period+1])
    avg_loss[rsi_period] = np.mean(loss[1:rsi_period+1])
    for i in range(rsi_period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index (14)
    chop_period = 14
    atr = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(1, n):
        atr[i] = (atr[i-1] * (chop_period-1) + tr[i]) / chop_period
    
    max_range = np.zeros(n)
    for i in range(chop_period-1, n):
        max_range[i] = np.max(high[i-chop_period+1:i+1]) - np.min(low[i-chop_period+1:i+1])
    
    chop = np.zeros(n)
    for i in range(chop_period-1, n):
        if max_range[i] != 0 and atr[i] != 0:
            chop[i] = 100 * np.log10(np.sum(tr[i-chop_period+1:i+1]) / max_range[i]) / np.log10(chop_period)
        else:
            chop[i] = 50
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(er_period, rsi_period, chop_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        weekly_trend = ema_1w_aligned[i]
        
        # Chop filter: avoid ranging markets (chop > 61.8)
        if chop_val > 61.8:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, weekly uptrend
            if price > kama_val and rsi_val > 50 and price > weekly_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, weekly downtrend
            elif price < kama_val and rsi_val < 50 and price < weekly_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below KAMA or RSI < 40
            if price < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above KAMA or RSI > 60
            if price > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_RSI_With_Chop_Filter"
timeframe = "1d"
leverage = 1.0