#!/usr/bin/env python3
# 4h_1d_RSI_Stochastic_Combo
# Hypothesis: Combine RSI(14) and Stochastic(14,3,3) on 4h timeframe with daily trend filter.
# Long when RSI < 30 and Stochastic %K crosses above %D (oversold bounce).
# Short when RSI > 70 and Stochastic %K crosses below %D (overbought rejection).
# Use daily EMA50 to filter trades: only long when price > daily EMA50, short when price < daily EMA50.
# This avoids counter-trend trades and works in both bull and bear markets by following the daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_RSI_Stochastic_Combo"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Calculate daily EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: RSI(14) ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / np.where(loss_ma > 0, loss_ma, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # === 4h: Stochastic(14,3,3) ===
    high = prices['high'].values
    low = prices['low'].values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / np.where(highest_high - lowest_low > 0, highest_high - lowest_low, np.nan)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Align daily EMA50 to 4h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after warmup
        # Get values
        close_val = close[i]
        rsi_val = rsi[i]
        k_val = k_percent[i]
        d_val = d_percent[i]
        ema50_val = ema50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(rsi_val) or np.isnan(k_val) or np.isnan(d_val) or 
            np.isnan(ema50_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RSI oversold and Stochastic bullish crossover, only in daily uptrend
            if (rsi_val < 30 and 
                k_percent[i-1] < d_percent[i-1] and  # Previous: K below D
                k_val > d_val and  # Current: K crosses above D
                close_val > ema50_val):  # Only long in daily uptrend
                signals[i] = 0.25
                position = 1
            # Short: RSI overbought and Stochastic bearish crossover, only in daily downtrend
            elif (rsi_val > 70 and 
                  k_percent[i-1] > d_percent[i-1] and  # Previous: K above D
                  k_val < d_val and  # Current: K crosses below D
                  close_val < ema50_val):  # Only short in daily downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought or Stochastic bearish crossover
            if (rsi_val > 70 or 
                (k_percent[i-1] > d_percent[i-1] and k_val < d_val)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold or Stochastic bullish crossover
            if (rsi_val < 30 or 
                (k_percent[i-1] < d_percent[i-1] and k_val > d_val)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals