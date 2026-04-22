#!/usr/bin/env python3

"""
12h_KAMA_Trend_RSI_MeanReversion
Trades mean-reversion at KAMA reversal points when price deviates from trend, filtered by RSI extremes and 1-week trend.
Designed for low trade frequency (15-30 trades/year) with clear trend/momentum confluence to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_len=10, fast_len=2, slow_len=30):
    """Calculate Kaufman Adaptive Moving Average (KAMA)."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily data for RSI and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI (14-period)
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Daily KAMA for trend
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(kama_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price deviation from KAMA (mean reversion signal)
        price_dev = (close[i] - kama_aligned[i]) / kama_aligned[i]
        
        if position == 0:
            # Long when price significantly below KAMA in uptrend with oversold RSI
            if price_dev < -0.015 and ema_34_1w_aligned[i] > close[i] and rsi_aligned[i] < 30:
                signals[i] = 0.25
                position = 1
            # Short when price significantly above KAMA in downtrend with overbought RSI
            elif price_dev > 0.015 and ema_34_1w_aligned[i] < close[i] and rsi_aligned[i] > 70:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to KAMA or RSI normalizes
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses above KAMA or RSI > 50
                if close[i] > kama_aligned[i] or rsi_aligned[i] > 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses below KAMA or RSI < 50
                if close[i] < kama_aligned[i] or rsi_aligned[i] < 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_Trend_RSI_MeanReversion"
timeframe = "12h"
leverage = 1.0