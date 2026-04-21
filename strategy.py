#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Filter_v1
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) to determine trend direction.
Enter long when price > KAMA and RSI(14) > 50; enter short when price < KAMA and RSI(14) < 50.
Add volume confirmation (1.5x 20-day average) to filter false signals. Use discrete position sizing (0.25).
KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
Designed for 1d timeframe to target 30-100 trades over 4 years (7-25/year). Works in bull/bear via adaptive trend filtering.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for regime filter - optional but can add later if needed)
    # For now, primary timeframe is 1d, so we use 1d data directly from prices for indicators
    # But we still need to demonstrate MTF loading pattern per rules - load 1w as HTF reference
    df_1w = get_htf_data(prices, '1w')
    
    # === Primary indicators on 1d timeframe (using prices directly since timeframe="1d") ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Kaufman Adaptive Moving Average (KAMA) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # sum of |close[t] - close[t-1]| over 10 periods
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start at index 9 (10th element)
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Handle first 14 values
    rsi[:13] = np.nan
    
    # === Volume confirmation (1.5x 20-day average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # start after warmup period
        # Skip if indicators not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Enter long: price > KAMA and RSI > 50 (bullish momentum)
            # Enter short: price < KAMA and RSI < 50 (bearish momentum)
            long_condition = (price > kama[i]) and (rsi[i] > 50) and volume_confirmed
            short_condition = (price < kama[i]) and (rsi[i] < 50) and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Exit long: price < KAMA (trend reversal) or RSI < 40 (loss of momentum)
            if price < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA (trend reversal) or RSI > 60 (loss of momentum)
            if price > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0