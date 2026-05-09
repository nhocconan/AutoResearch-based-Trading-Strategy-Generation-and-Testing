#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_KAMA_Trend_RSI_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    """
    4h KAMA trend filter with RSI momentum and chop filter.
    - Uses KAMA to determine trend direction (adaptive to market conditions)
    - RSI for momentum confirmation (avoid overbought/oversold extremes)
    - Choppiness index to filter ranging markets (only trade when trending)
    - Target: 20-50 trades/year on 4h timeframe
    """
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array lengths
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start after 10 periods
    for i in range(10, len(close_1d)):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    kama_4h = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on 4h close (14 periods)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index on 1d (14 periods)
    atr_1d = pd.Series(np.maximum(
        np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])),
        np.abs(low_1d[1:] - close_1d[:-1])
    )).rolling(window=14, min_periods=14).mean().values
    atr_1d = np.concatenate([np.full(14, np.nan), atr_1d])
    
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    chop = np.where(
        (max_high_1d - min_low_1d) != 0,
        100 * np.log10(np.sum(atr_1d) / (max_high_1d - min_low_1d)) / np.log10(14),
        50
    )
    chop = np.concatenate([np.full(14, np.nan), chop])
    
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Weekly EMA for higher timeframe trend
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_4h[i]) or np.isnan(rsi[i]) or np.isnan(chop_4h[i]) or 
            np.isnan(ema50_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Chop filter: only trade when trending (CHOP < 38.2)
        trending = chop_4h[i] < 38.2
        
        if position == 0:
            # Long: Price above KAMA (uptrend), RSI > 50 (momentum), above weekly EMA (HTF trend)
            if (close[i] > kama_4h[i] and rsi[i] > 50 and 
                close[i] > ema50_4h[i] and trending):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend), RSI < 50 (momentum), below weekly EMA (HTF trend)
            elif (close[i] < kama_4h[i] and rsi[i] < 50 and 
                  close[i] < ema50_4h[i] and trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below KAMA OR RSI < 40 (loss of momentum)
            if close[i] < kama_4h[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above KAMA OR RSI > 60 (loss of momentum)
            if close[i] > kama_4h[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals