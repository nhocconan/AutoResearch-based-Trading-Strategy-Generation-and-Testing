#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA + RSI + Chop filter strategy
# Uses 1-day KAMA to establish trend direction (bullish when price > KAMA)
# Combines with RSI(14) for entry timing (oversold in uptrend, overbought in downtrend)
# Uses weekly Choppiness Index to filter ranging markets (avoid trading when CHOP > 61.8)
# Designed to work in both bull and bear markets by using adaptive trend filter
# Target: 10-25 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for chop filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Choppiness Index
    chop_len = 14
    if len(df_1w) >= chop_len:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # ATR
        atr_1w = pd.Series(tr).ewm(span=chop_len, adjust=False, min_periods=chop_len).mean().values
        
        # Highest high and lowest low over chop_len periods
        highest_high = pd.Series(high_1w).rolling(window=chop_len, min_periods=chop_len).max().values
        lowest_low = pd.Series(low_1w).rolling(window=chop_len, min_periods=chop_len).min().values
        
        # Chop calculation
        sum_tr = pd.Series(tr).rolling(window=chop_len, min_periods=chop_len).sum().values
        chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(chop_len)
        chop[highest_high == lowest_low] = 100  # Avoid division by zero
        chop_align = align_htf_to_ltf(prices, df_1w, chop)
    else:
        chop_align = np.full(n, np.nan)
    
    # Daily KAMA calculation
    kama_len = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    if len(close) >= kama_len:
        # Efficiency Ratio
        change = np.abs(np.diff(close, kama_len))
        change = np.concatenate([np.full(kama_len, np.nan), change])
        
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        volatility = pd.Series(volatility).rolling(window=kama_len, min_periods=kama_len).sum().values
        
        er = np.where(volatility != 0, change / volatility, 0)
        
        # Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA
        kama = np.full_like(close, np.nan)
        kama[kama_len] = close[kama_len]
        
        for i in range(kama_len + 1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    else:
        kama = np.full(n, np.nan)
    
    # Daily RSI
    rsi_len = 14
    if len(close) >= rsi_len:
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(kama_len*2, rsi_len, chop_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop_align[i])):
            signals[i] = 0.0
            continue
        
        # Chop filter: only trade when market is not too choppy (trending)
        # CHOP > 61.8 indicates ranging market, avoid trading
        not_choppy = chop_align[i] <= 61.8
        
        if position == 0:
            # Long signal: price above KAMA (uptrend) and RSI oversold
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                not_choppy):
                position = 1
                signals[i] = position_size
            # Short signal: price below KAMA (downtrend) and RSI overbought
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  not_choppy):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA or RSI overbought
            if (close[i] < kama[i] or 
                rsi[i] > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA or RSI oversold
            if (close[i] > kama[i] or 
                rsi[i] < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "daily_kama_rsi_chop_filter_v1"
timeframe = "1d"
leverage = 1.0