#!/usr/bin/env python3
"""
1d_KAMA_Trend_RSI_ChopFilter_v1
Hypothesis: On 1d timeframe, use Kaufman Adaptive Moving Average (KAMA) for trend direction, combined with RSI(14) for momentum and Choppiness Index(14) for regime filtering. Enter long when KAMA trending up, RSI > 50, and CHOP < 38.2 (trending regime). Enter short when KAMA trending down, RSI < 50, and CHOP < 38.2. Uses ATR(14) stoploss at 2.0x ATR. Discrete sizing at 0.25 to limit fee drag. Target: 7-25 trades/year on BTC/ETH/SOL.
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
    
    # Get 1w data for trend filter (more stable than 1d)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w KAMA for trend filter
    close_1w = df_1w['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period volatility
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close_1w, np.nan)
    kama[9] = close_1w[9]  # Start after first 10 periods
    for i in range(10, len(close_1w)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # Calculate 1d RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1d Choppiness Index(14)
    chop_period = 14
    atr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    atr[0] = high[0] - low[0]
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(chop_period)
    # Avoid division by zero
    chop = np.where((max_high - min_low) != 0, chop, 50)
    
    # Align HTF indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    
    # ATR for stoploss calculation (1d ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of KAMA (10+30), RSI (14), CHOP (14), ATR (14)
    start_idx = max(40, 14, 14, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(chop[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        chop_val = chop[i]
        close_val = close[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: KAMA trending up (price > KAMA), RSI > 50, CHOP < 38.2 (trending)
            long_signal = (close_val > kama_val) and (rsi_val > 50) and (chop_val < 38.2)
            
            # Short: KAMA trending down (price < KAMA), RSI < 50, CHOP < 38.2 (trending)
            short_signal = (close_val < kama_val) and (rsi_val < 50) and (chop_val < 38.2)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price crosses below KAMA OR ATR stoploss (2.0*ATR below entry)
            if (close_val < kama_val) or (close_val < entry_price - 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above KAMA OR ATR stoploss (2.0*ATR above entry)
            if (close_val > kama_val) or (close_val > entry_price + 2.0 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0