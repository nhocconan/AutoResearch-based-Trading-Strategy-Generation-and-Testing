#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction + RSI + Chop regime
# Uses KAMA (Kaufman Adaptive Moving Average) to identify trend direction with adaptive smoothing
# RSI(14) for momentum confirmation, avoiding overbought/oversold extremes
# Choppiness Index(14) to filter ranging markets (CHOP > 61.8) and only trade in trending markets (CHOP < 38.2)
# Weekly trend filter from 1w EMA34 to avoid counter-trend trades
# Designed for low-frequency, high-conviction trades in both bull and bear markets
# Target: 30-100 total trades over 4 years = 7-25/year

name = "1d_KAMA_RSI_Chop_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate KAMA(10) on daily data
    # Efficiency Ratio (ER) = abs(close - close[10]) / sum(abs(close - close[1])) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of absolute daily changes
    # Pad arrays to match length
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad RSI to match length
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Calculate Choppiness Index(14)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop = 100 * log10(sum_tr / (hh - ll)) / log10(14)
    range_hl = hh - ll
    chop = np.where(range_hl > 0, 100 * np.log10(sum_tr / range_hl) / np.log10(14), 50)
    # Pad to match length
    chop = np.concatenate([np.full(13, np.nan), chop])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: price > KAMA (uptrend), RSI > 50 (bullish momentum), Chop < 38.2 (trending), price > weekly EMA34 (uptrend)
            if (close[i] > kama_val and 
                rsi_val > 50 and 
                chop_val < 38.2 and 
                close[i] > ema34_1w_val):
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (downtrend), RSI < 50 (bearish momentum), Chop < 38.2 (trending), price < weekly EMA34 (downtrend)
            elif (close[i] < kama_val and 
                  rsi_val < 50 and 
                  chop_val < 38.2 and 
                  close[i] < ema34_1w_val):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < KAMA OR Chop > 61.8 (ranging) OR RSI < 40 (losing momentum)
            if (close[i] < kama_val or 
                chop_val > 61.8 or 
                rsi_val < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > KAMA OR Chop > 61.8 (ranging) OR RSI > 60 (losing momentum)
            if (close[i] > kama_val or 
                chop_val > 61.8 or 
                rsi_val > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals