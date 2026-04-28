#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter
Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, RSI(14) for momentum, and Choppiness Index (CHOP) to filter ranging markets. Enters long when KAMA upward, RSI > 50, and CHOP < 38.2 (trending); short when KAMA downward, RSI < 50, and CHOP < 38.2. Avoids ranging markets (CHOP > 61.8). Uses weekly trend filter to avoid counter-trend trades. Targets 10-20 trades/year for low frequency and high conviction.
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate KAMA(10) for trend
    def calculate_kama(close, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        # Fix for array operations
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)
        er = np.zeros_like(close)
        for i in range(1, len(close)):
            num = np.abs(close[i] - close[i-length]) if i >= length else 0
            den = np.sum(np.abs(np.diff(close[i-length:i+1]))) if i >= length else 1
            er[i] = num / den if den != 0 else 0
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10)
    
    # Calculate RSI(14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate Choppiness Index (CHOP) with period 14
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
            atr[i] = (atr[i-1] * (length-1) + tr) / length if i >= 1 else tr
        sum_tr = pd.Series(atr).rolling(window=length, min_periods=length).sum().values
        highest_high = pd.Series(high).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(low).rolling(window=length, min_periods=length).min().values
        chop = 100 * np.log10(sum_tr / (highest_high - lowest_low)) / np.log10(length)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend: slope of KAMA
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # RSI momentum
        rsi_bull = rsi[i] > 50
        rsi_bear = rsi[i] < 50
        
        # Chop filter: trending market
        chop_trending = chop[i] < 38.2
        chop_ranging = chop[i] > 61.8
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions
        long_entry = kama_up and rsi_bull and chop_trending and weekly_uptrend
        short_entry = kama_down and rsi_bear and chop_trending and weekly_downtrend
        
        # Exit conditions: opposite signal or chop becomes ranging
        long_exit = not (kama_up and rsi_bull) or chop_ranging
        short_exit = not (kama_down and rsi_bear) or chop_ranging
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0