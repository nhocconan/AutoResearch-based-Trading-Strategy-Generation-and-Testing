#!/usr/bin/env python3
"""
1d_SMA_Trend_RSI_Confirmation_v1
Hypothesis: Use daily SMA20 for trend direction and daily RSI14 for momentum confirmation.
Go long when price > SMA20 AND RSI > 55, short when price < SMA20 AND RSI < 45.
Add weekly trend filter using SMA50: only take longs when weekly close > weekly SMA50,
and shorts when weekly close < weekly SMA50.
This reduces false signals in ranging markets and captures major trends.
Target: 10-20 trades/year by requiring multiple confluence factors.
Works in bull markets via trend following and in bear via short signals with weekly filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get daily data for SMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily SMA20
    sma_period = 20
    sma_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= sma_period:
        for i in range(sma_period, len(close_1d)):
            sma_1d[i] = np.mean(close_1d[i - sma_period:i])
    
    # Daily RSI14
    rsi_period = 14
    rsi_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly SMA50
    sma50_1w = np.full_like(close_1w, np.nan)
    sma50_period = 50
    if len(close_1w) >= sma50_period:
        for i in range(sma50_period, len(close_1w)):
            sma50_1w[i] = np.mean(close_1w[i - sma50_period:i])
    
    # Align all indicators to daily timeframe
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(sma_period, rsi_period + 1, sma50_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > daily SMA20 AND RSI > 55 AND weekly close > weekly SMA50
            if close[i] > sma_1d_aligned[i] and rsi_1d_aligned[i] > 55 and close[i] > sma50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < daily SMA20 AND RSI < 45 AND weekly close < weekly SMA50
            elif close[i] < sma_1d_aligned[i] and rsi_1d_aligned[i] < 45 and close[i] < sma50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < daily SMA20 OR RSI < 40
            if close[i] < sma_1d_aligned[i] or rsi_1d_aligned[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > daily SMA20 OR RSI > 60
            if close[i] > sma_1d_aligned[i] or rsi_1d_aligned[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_SMA_Trend_RSI_Confirmation_v1"
timeframe = "1d"
leverage = 1.0