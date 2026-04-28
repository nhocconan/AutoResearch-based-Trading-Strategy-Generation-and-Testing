#!/usr/bin/env python3
"""
1d_KAMA_Direction_1wTrend_RSI_Filter
Hypothesis: Daily Kaufman Adaptive Moving Average (KAMA) captures trend direction while adapting to market noise. Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend trades. RSI(14) > 50 for longs and < 50 for shorts adds momentum confirmation. This combination should work in both bull and bear markets by avoiding whipsaw during ranging periods. Target: 10-20 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close, prepend=close[0]))
    
    # Efficiency ratio
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i < er_length:
            er[i] = np.nan
        else:
            price_change = np.abs(close[i] - close[i-er_length])
            sum_volatility = np.sum(volatility[i-er_length+1:i+1])
            er[i] = price_change / sum_volatility if sum_volatility != 0 else 0
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    wkama = calculate_kama(close_1w, er_length=10, fast_sc=2, slow_sc=30)
    wkama_aligned = align_htf_to_ltf(prices, df_1w, wkama)
    
    # Calculate daily KAMA for direction
    dkama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dkama[i]) or np.isnan(wkama_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction
        price_above_dkama = close[i] > dkama[i]
        price_below_dkama = close[i] < dkama[i]
        
        # Weekly trend filter
        price_above_wkama = close[i] > wkama_aligned[i]
        price_below_wkama = close[i] < wkama_aligned[i]
        
        # RSI filter
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry conditions
        long_entry = price_above_dkama and price_above_wkama and rsi_bullish
        short_entry = price_below_dkama and price_below_wkama and rsi_bearish
        
        # Exit conditions - reverse on opposite signal
        long_exit = price_below_dkama or price_below_wkama or not rsi_bullish
        short_exit = price_above_dkama or price_above_wkama or not rsi_bearish
        
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

name = "1d_KAMA_Direction_1wTrend_RSI_Filter"
timeframe = "1d"
leverage = 1.0