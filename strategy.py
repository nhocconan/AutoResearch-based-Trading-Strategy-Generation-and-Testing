#!/usr/bin/env python3
"""
1d_KAMA_RSI_Chop_Filter_v1
Hypothesis: 1-day KAMA identifies trend direction, RSI filters pullbacks,
and Choppiness index avoids choppy regimes. Works in bull (trend following) and bear (mean reversion in range) markets.
Designed for ~15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    if len(close) < er_period:
        return np.full(len(close), np.nan)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.zeros(len(close))
    er[er_period-1:] = change / np.where(volatility[er_period-1:] == 0, 1, volatility[er_period-1:])
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index."""
    if len(close) < period + 1:
        return np.full(len(close), np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(len(close))
    avg_loss = np.zeros(len(close))
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    if len(close) < period:
        return np.full(len(close), np.nan)
    
    atr = np.zeros(len(close))
    for i in range(1, len(close)):
        atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR over period
    atr_sum = np.zeros(len(close))
    for i in range(period, len(close)):
        atr_sum[i] = np.sum(atr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros(len(close))
    ll = np.zeros(len(close))
    for i in range(period-1, len(close)):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness
    chop = np.zeros(len(close))
    for i in range(period-1, len(close)):
        if atr_sum[i] > 0 and hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for higher timeframe filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate indicators
    kama = calculate_kama(close, er_period=10, fast=2, slow=30)
    rsi = calculate_rsi(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 1-week EMA34 for trend filter
    if len(close_1w) >= 34:
        ema_34_1w = np.zeros(len(close_1w))
        ema_34_1w[0] = close_1w[0]
        alpha = 2 / (34 + 1)
        for i in range(1, len(close_1w)):
            ema_34_1w[i] = ema_34_1w[i-1] + alpha * (close_1w[i] - ema_34_1w[i-1])
    else:
        ema_34_1w = np.full(len(close_1w), np.nan)
    
    # Align 1-week EMA to daily
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Chop filter: avoid choppy markets (chop > 61.8) or strong trends (chop < 38.2)
        # We trade in moderate chop: 38.2 <= chop <= 61.8
        chop_filter = (chop[i] >= 38.2) and (chop[i] <= 61.8)
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI not overbought, in chop zone, weekly uptrend
            if (close[i] > kama[i] and rsi[i] < 70 and chop_filter and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend), RSI not oversold, in chop zone, weekly downtrend
            elif (close[i] < kama[i] and rsi[i] > 30 and chop_filter and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below KAMA or RSI overbought
            if close[i] < kama[i] or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above KAMA or RSI oversold
            if close[i] > kama[i] or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_Chop_Filter_v1"
timeframe = "1d"
leverage = 1.0