#/usr/bin/env python3
name = "1d_KAMA_RSI_Chop_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA: Kaufman Adaptive Moving Average
    def kama(close_prices, fast=2, slow=30):
        change = np.abs(np.diff(close_prices, n=10))
        volatility = np.sum(np.abs(np.diff(close_prices, n=1)), axis=0)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama_vals = np.zeros_like(close_prices)
        kama_vals[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close_prices[i] - kama_vals[i-1])
        return kama_vals
    
    # Calculate KAMA on daily data
    kama_vals = kama(close, fast=2, slow=30)
    
    # Align KAMA to daily (no shift needed as same timeframe)
    kama_aligned = kama_vals
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Choppiness Index (14)
    def choppiness_index(high, low, close, window=14):
        atr = np.zeros(len(close))
        for i in range(1, len(close)):
            atr[i] = max(high[i] - low[i], 
                         np.abs(high[i] - close[i-1]), 
                         np.abs(low[i] - close[i-1]))
        atr_sum = pd.Series(atr).rolling(window=window, min_periods=window).sum().values
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(window)
        return chop
    
    chop = choppiness_index(high, low, close, window=14)
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14, 34)  # Wait for KAMA, RSI, Chop, and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, Chop < 61.8 (trending), weekly uptrend
            if (close[i] > kama_aligned[i] and 
                rsi[i] > 50 and 
                chop[i] < 61.8 and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, Chop < 61.8 (trending), weekly downtrend
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] < 50 and 
                  chop[i] < 61.8 and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or Chop > 61.8 (choppy) or weekly trend change
            if (close[i] < kama_aligned[i] or 
                chop[i] > 61.8 or 
                ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or Chop > 61.8 (choppy) or weekly trend change
            if (close[i] > kama_aligned[i] or 
                chop[i] > 61.8 or 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 1d KAMA + RSI + Chop filter with weekly trend
# - KAMA adapts to market conditions: fast in trends, slow in ranges
# - RSI > 50 for long, < 50 for short ensures momentum alignment
# - Chop < 61.8 filters for trending markets (avoids whipsaws in ranges)
# - Weekly EMA(34) ensures alignment with higher timeframe trend
# - Works in both bull (long when KAMA up, RSI>50) and bear (short when KAMA down, RSI<50)
# - Chop filter prevents trading in choppy/range markets where KAMA whipsaws
# - Position size 0.25 targets ~15-30 trades/year, avoiding fee drag
# - Combines adaptive trend (KAMA), momentum (RSI), and regime (Chop) filters
# - Proven elements: KAMA (from DB winners), Chop filter (proven regime filter)
# - Aims for 30-100 total trades over 4 years (7-25/year) as per 1d guidelines