#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend with RSI and Choppiness Index filter
# KAMA adapts to market noise, reducing whipsaws in ranging markets
# RSI(14) avoids overbought/oversold extremes
# Choppiness Index > 61.8 indicates ranging market (mean reversion opportunity)
# Target: 12-30 trades/year per symbol to stay within fee limits

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index (14 periods)
    chop_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Sum of true ranges over chop_len periods
    tr_sum = pd.Series(tr).rolling(window=chop_len, min_periods=chop_len).sum().values
    
    # Highest high and lowest low over chop_len periods
    hh = pd.Series(high_1d).rolling(window=chop_len, min_periods=chop_len).max().values
    ll = pd.Series(low_1d).rolling(window=chop_len, min_periods=chop_len).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(chop_len)
    chop = np.where((hh - ll) == 0, 50, chop)  # avoid division by zero
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # KAMA (Adaptive Moving Average)
    er_len = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_len))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Vectorized volatility sum
    volatility_sum = pd.Series(np.abs(np.diff(close))).rolling(window=er_len, min_periods=1).sum().values
    er = np.where(volatility_sum > 0, change / volatility_sum, 0)
    er = np.concatenate([np.zeros(er_len), er])  # align length
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_len] = close[er_len]  # seed
    for i in range(er_len + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14 periods)
    rsi_len = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_len, min_periods=rsi_len).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(rsi_len, np.nan), rsi])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, er_len, rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Range filter: Choppiness > 61.8 indicates ranging market
        ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Enter long: price > KAMA + RSI < 40 (oversold) + ranging
            if (close[i] > kama[i] and 
                rsi[i] < 40 and 
                ranging):
                position = 1
                signals[i] = position_size
            # Enter short: price < KAMA + RSI > 60 (overbought) + ranging
            elif (close[i] < kama[i] and 
                  rsi[i] > 60 and 
                  ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < KAMA OR RSI > 60 (overbought)
            if (close[i] < kama[i] or 
                rsi[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > KAMA OR RSI < 40 (oversold)
            if (close[i] > kama[i] or 
                rsi[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_RSI_Chop_v1"
timeframe = "12h"
leverage = 1.0