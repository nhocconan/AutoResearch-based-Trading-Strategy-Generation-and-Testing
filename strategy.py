#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter_v1
Hypothesis: Kaufman Adaptive Moving Average (KAMA) determines trend direction on daily timeframe.
Enter long when KAMA slope > 0 and RSI(14) > 50; short when KAMA slope < 0 and RSI(14) < 50.
Add choppiness index regime filter: only trade when CHOP(14) < 61.8 (trending market).
Uses 1-week EMA50 as higher timeframe trend filter to avoid counter-trend trades.
Discrete position sizing (0.0, ±0.25) to minimize fee churn. Target: 20-80 total trades over 4 years.
Works in bull markets via trend following and in bear markets via short signals aligned with weekly trend.
"""

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
    
    # === PRIMARY INDICATORS (1d timeframe) ===
    # KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_length=10, fast_ma=2, slow_ma=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) == 0 else pd.Series(close).diff().abs().rolling(er_length).sum().values
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, er_length=10, fast_ma=2, slow_ma=30)
    kama_slope = np.diff(kama, prepend=kama[0])  # daily change in KAMA
    
    # RSI(14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Choppiness Index (CHOP) - regime filter
    def calculate_chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        
        max_high = pd.Series(high).rolling(length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(length, min_periods=length).min().values
        
        chop = np.where(atr != 0, 100 * np.log10((max_high - min_low) / (atr * length)) / np.log10(length), 50)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop < 61.8  # trending regime
    
    # === MTF: WEEKLY TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === SIGNAL LOGIC ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of all lookbacks
    start_idx = max(30, 14, 14, 50) + 5
    
    base_size = 0.25
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or np.isnan(ema_50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long conditions: KAMA rising + RSI > 50 + choppy filter + price above weekly EMA50
        long_condition = (kama_slope[i] > 0 and 
                         rsi[i] > 50 and 
                         chop_filter[i] and 
                         close[i] > ema_50_1w_aligned[i])
        
        # Short conditions: KAMA falling + RSI < 50 + choppy filter + price below weekly EMA50
        short_condition = (kama_slope[i] < 0 and 
                          rsi[i] < 50 and 
                          chop_filter[i] and 
                          close[i] < ema_50_1w_aligned[i])
        
        # Entry logic
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        # Exit logic: opposite signal or loss of trend alignment
        elif position == 1 and (kama_slope[i] <= 0 or rsi[i] <= 50 or not chop_filter[i] or close[i] <= ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (kama_slope[i] >= 0 or rsi[i] >= 50 or not chop_filter[i] or close[i] >= ema_50_1w_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0