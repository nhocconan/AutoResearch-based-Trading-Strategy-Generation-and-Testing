#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA with weekly trend filter and volume confirmation
# KAMA adapts to market noise, reducing false signals in choppy markets
# Weekly trend filter ensures we only trade in the direction of higher timeframe momentum
# Volume confirmation adds conviction to signals
# Target: 15-25 trades/year per symbol to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(34) for trend direction
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate KAMA on daily data
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    er = np.zeros_like(close)
    er[10:] = change[10:] / volatility[10:]
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 34, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_weekly_aligned[i]) or 
            np.isnan(kama[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_weekly_aligned[i]
        downtrend = close[i] < ema_weekly_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price above KAMA + uptrend + volume
            if (close[i] > kama[i] and 
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price below KAMA + downtrend + volume
            elif (close[i] < kama[i] and 
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "daily_kama_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0