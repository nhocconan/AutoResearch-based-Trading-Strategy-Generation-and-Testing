#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI(14) with weekly trend filter and volume confirmation
# KAMA adapts to market noise, reducing false signals in choppy markets
# RSI(14) > 50 for long, < 50 for short ensures momentum alignment
# Weekly EMA20 filter ensures trades align with higher timeframe trend
# Volume > 1.3x 20-period average confirms institutional participation
# Designed for 1d timeframe with selective entries to avoid overtrading
# Target: 7-25 trades per year per symbol (28-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on weekly timeframe for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate KAMA ( Kaufman Adaptive Moving Average )
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    er = np.zeros_like(change)
    er[10:] = change[10:] / (volatility[10:] + 1e-10)  # avoid division by zero
    
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup period
        # Skip if NaN in indicators
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or \
           np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        
        # KAMA direction
        kama_up = close[i] > kama[i]
        kama_down = close[i] < kama[i]
        
        # RSI condition
        rsi_long = rsi[i] > 50
        rsi_short = rsi[i] < 50
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price > KAMA + RSI > 50 + weekly uptrend + volume
            long_signal = kama_up and rsi_long and is_uptrend and has_volume
            
            # Short entry: price < KAMA + RSI < 50 + weekly downtrend + volume
            short_signal = kama_down and rsi_short and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price < KAMA or RSI < 40
            exit_signal = (close[i] < kama[i]) or (rsi[i] < 40)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > KAMA or RSI > 60
            exit_signal = (close[i] > kama[i]) or (rsi[i] > 60)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_RSI_WeeklyTrendFilter_Volume"
timeframe = "1d"
leverage = 1.0