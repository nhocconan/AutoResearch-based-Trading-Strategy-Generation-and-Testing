#!/usr/bin/env python3

"""
Hypothesis: 1-day KAMA + RSI + Choppiness regime filter.
Trades in direction of KAMA trend when RSI shows momentum and market is not choppy.
Uses weekly EMA trend filter for multi-timeframe confirmation.
Designed for low trade frequency (7-25/year) to minimize fee drag and work in both bull and bear markets
by adapting to trending vs ranging regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    
    # Handle first element
    change[0] = 0
    
    er = np.zeros_like(close)
    for i in range(len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_rsi(close, length=14):
    """Calculate Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[length] = np.mean(gain[1:length+1])
    avg_loss[length] = np.mean(loss[1:length+1])
    
    for i in range(length+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_choppiness(high, low, close, length=14):
    """Calculate Choppiness Index."""
    atr = np.zeros_like(close)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # First TR
    tr[0] = tr1[0]
    
    # Sum of TR over period
    atr_sum = np.zeros_like(close)
    for i in range(length, len(close)):
        atr_sum[i] = np.sum(tr[i-length+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros_like(close)
    ll = np.zeros_like(close)
    for i in range(length-1, len(close)):
        hh[i] = np.max(high[i-length+1:i+1])
        ll[i] = np.min(low[i-length+1:i+1])
    
    chop = np.zeros_like(close)
    for i in range(length-1, len(close)):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(length)
        else:
            chop[i] = 50
    
    return chop

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
    
    # Weekly EMA for trend filter (34-period)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily KAMA for trend (10-period ER, 2/30 SC)
    kama = calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Daily RSI for momentum (14-period)
    rsi = calculate_rsi(close, length=14)
    
    # Daily Choppiness for regime filter (14-period)
    chop = calculate_choppiness(high, low, close, length=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when not choppy (Choppiness < 61.8)
        not_choppy = chop[i] < 61.8
        
        if position == 0 and not_choppy:
            # Long: price above KAMA, RSI > 50 (bullish momentum), weekly uptrend
            if close[i] > kama[i] and rsi[i] > 50 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50 (bearish momentum), weekly downtrend
            elif close[i] < kama[i] and rsi[i] < 50 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite condition or choppy market
            exit_signal = False
            
            if position == 1:
                # Exit long: price below KAMA or RSI < 40 or market becomes choppy
                if close[i] < kama[i] or rsi[i] < 40 or chop[i] >= 61.8:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above KAMA or RSI > 60 or market becomes choppy
                if close[i] > kama[i] or rsi[i] > 60 or chop[i] >= 61.8:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_1wEMA34_Trend"
timeframe = "1d"
leverage = 1.0