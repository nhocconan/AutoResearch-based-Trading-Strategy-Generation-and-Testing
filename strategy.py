#!/usr/bin/env python3
"""
1h_SMA100_RSI20_Stoch50_Trend_v1
Trend-following strategy on 1h timeframe with 100-period SMA for trend direction,
RSI(14) for momentum, and Stochastic(14,3,3) for timing. Uses 4h EMA50 for higher timeframe filter.
Trades only during 08-20 UTC to avoid low-volume periods.
Target: 60-150 total trades over 4 years = 15-37/year.
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
    
    # === 100-period SMA for trend direction ===
    sma100 = np.full(n, np.nan)
    for i in range(100, n):
        sma100[i] = np.mean(close[i-99:i+1])
    
    # === RSI(14) for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Stochastic(14,3,3) for timing ===
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    for i in range(14, n):
        lowest_low[i] = np.min(low[i-13:i+1])
        highest_high[i] = np.max(high[i-13:i+1])
    
    stoch_raw = np.where((highest_high - lowest_low) != 0, 
                         (close - lowest_low) / (highest_high - lowest_low) * 100, 50)
    
    # Smooth %K with 3-period SMA
    stoch_k = np.full(n, np.nan)
    for i in range(16, n):
        stoch_k[i] = np.mean(stoch_raw[i-2:i+1])
    
    # Smooth %D with 3-period SMA
    stoch_d = np.full(n, np.nan)
    for i in range(19, n):
        stoch_d[i] = np.mean(stoch_k[i-2:i+1])
    
    # === 4h EMA50 for higher timeframe filter ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    ema50_4h = np.full(len(close_4h), np.nan)
    for i in range(50, len(close_4h)):
        if i == 50:
            ema50_4h[i] = np.mean(close_4h[0:51])
        else:
            ema50_4h[i] = (close_4h[i] * 2 / (50 + 1)) + (ema50_4h[i-1] * (49 / (50 + 1)))
    
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(sma100[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(stoch_d[i]) or 
            np.isnan(ema50_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Check session: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price above SMA100 (uptrend), RSI > 50 (momentum), Stoch rising from oversold
            if (close[i] > sma100[i] and 
                rsi[i] > 50 and 
                stoch_d[i] < 30 and 
                stoch_d[i] > stoch_d[i-1] and  # Stoch rising
                close[i] > ema50_4h_aligned[i]):  # 4h uptrend filter
                signals[i] = 0.20
                position = 1
                continue
            # Short: price below SMA100 (downtrend), RSI < 50 (momentum), Stoch falling from overbought
            elif (close[i] < sma100[i] and 
                  rsi[i] < 50 and 
                  stoch_d[i] > 70 and 
                  stoch_d[i] < stoch_d[i-1] and  # Stoch falling
                  close[i] < ema50_4h_aligned[i]):  # 4h downtrend filter
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price crosses below SMA100 OR RSI < 40 OR Stoch > 80
            if (close[i] < sma100[i] or 
                rsi[i] < 40 or 
                stoch_d[i] > 80):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above SMA100 OR RSI > 60 OR Stoch < 20
            if (close[i] > sma100[i] or 
                rsi[i] > 60 or 
                stoch_d[i] < 20):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_SMA100_RSI20_Stoch50_Trend_v1"
timeframe = "1h"
leverage = 1.0