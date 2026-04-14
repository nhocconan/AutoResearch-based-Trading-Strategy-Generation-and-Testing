#!/usr/bin/env python3
"""
12h_1d_KAMA_RSI_Trend_v1
Hypothesis: On 12h timeframe, use KAMA direction for primary trend filter, RSI(14) for momentum confirmation, and 1d EMA50 for higher-timeframe trend alignment. Enter long when KAMA is rising (bullish), RSI > 55, and price > 1d EMA50. Enter short when KAMA is falling (bearish), RSI < 45, and price < 1d EMA50. This captures momentum in the direction of the higher timeframe trend while avoiding choppy markets. Designed for 12h to target ~20-30 trades/year, reducing fee drag and improving generalization to bear markets (2025+).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA on 1d data
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_50_1d[i-1] * (49 / (50 + 1)))
    
    # Calculate KAMA on 12h data (using close prices)
    # ER (Efficiency Ratio) and SC (Smoothing Constant) with fast=2, slow=30
    def calculate_kama(close, slow=30, fast=2):
        n = len(close)
        kama = np.full(n, np.nan)
        if n < 2:
            return kama
        # Direction
        direction = np.abs(close - np.roll(close, 1))
        # Volatility (sum of absolute changes)
        volatility = np.sum(np.abs(np.diff(close)))
        # Avoid division by zero
        er = np.where(volatility != 0, direction / volatility, 0)
        # Smoothing constant
        sc = np.square(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1))
        # Initialize
        kama[0] = close[0]
        for i in range(1, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # We'll compute KAMA inside loop using available data up to i
    # But to avoid look-ahead, we'll precompute what we can and use expanding window logic
    # Instead, we compute KAMA incrementally in the loop
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # KAMA state variables (updated each bar)
    kama_prev = close[0]
    er_prev = 0.0
    # Constants
    fast_sc = 2/(2+1)   # 0.6667
    slow_sc = 2/(30+1)  # 0.0645
    
    for i in range(1, n):
        # Update KAMA with data up to i
        if i == 1:
            direction = abs(close[i] - close[i-1])
            volatility = direction  # first period
            er = 1.0 if volatility != 0 else 0.0
        else:
            direction = abs(close[i] - close[i-1])
            volatility = np.sum(np.abs(np.diff(close[:i+1])))  # sum from 0 to i
            er = direction / volatility if volatility != 0 else 0.0
        
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama_curr = kama_prev + sc * (close[i] - kama_prev)
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_curr > kama_prev
        kama_falling = kama_curr < kama_prev
        
        # RSI(14) calculation
        if i >= 14:
            # Calculate RSI using last 14 periods
            changes = np.diff(close[i-13:i+1])  # indices i-13 to i
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi = 50.0  # neutral until enough data
        
        # Get aligned 1d EMA50 value
        ema_50 = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: KAMA rising, RSI > 55, price > 1d EMA50
            if kama_rising and rsi > 55 and not np.isnan(ema_50) and close[i] > ema_50:
                position = 1
                signals[i] = position_size
            # Short entry: KAMA falling, RSI < 45, price < 1d EMA50
            elif kama_falling and rsi < 45 and not np.isnan(ema_50) and close[i] < ema_50:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns falling (trend change)
            if kama_falling:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA turns rising (trend change)
            if kama_rising:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        
        # Update for next iteration
        kama_prev = kama_curr
        er_prev = er
    
    return signals

name = "12h_1d_KAMA_RSI_Trend_v1"
timeframe = "12h"
leverage = 1.0