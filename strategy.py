#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_RSI_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise - follows trend in trending markets, stays flat in ranging markets.
Combined with RSI(14) for overbought/oversold conditions and 1d trend filter for multi-timeframe confirmation.
Designed for 12h timeframe to capture medium-term swings while avoiding whipsaws in chop.
Works in bull markets via trend following, in bear markets via mean reversion at extremes.
Target: 15-25 trades/year on 12h timeframe with disciplined entry conditions.
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
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    def calculate_kama(close, period=10, fast=2, slow=30):
        """Calculate Kaufman Adaptive Moving Average"""
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        
        # Avoid division by zero
        er = np.zeros_like(change, dtype=float)
        mask = volatility != 0
        er[mask] = change[mask] / volatility[mask]
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period] = close[period]  # Initialize
        
        for i in range(period + 1, len(close)):
            if not np.isnan(sc[i-period]):
                kama[i] = kama[i-1] + sc[i-period] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
                
        return kama
    
    # RSI (Relative Strength Index)
    def calculate_rsi(close, period=14):
        """Calculate Relative Strength Index"""
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        
        # Initial average
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
        
        # Wilder's smoothing
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Calculate indicators
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    
    # 1d EMA50 trend filter (from higher timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMA50 for 1d data
    k_ema = 2 / (50 + 1)
    for i in range(50, len(close_1d)):
        if i == 50:
            ema50_1d[i] = np.mean(close_1d[0:51])
        else:
            ema50_1d[i] = close_1d[i] * k_ema + ema50_1d[i-1] * (1 - k_ema)
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) and RSI not overbought
            if (close[i] > kama[i] and rsi[i] < 70 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) and RSI not oversold
            elif (close[i] < kama[i] and rsi[i] > 30 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought
            if (close[i] < kama[i] or rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold
            if (close[i] > kama[i] or rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0