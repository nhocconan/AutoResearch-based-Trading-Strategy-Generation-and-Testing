#!/usr/bin/env python3
# 1d_1w_KAMA_Direction_RSI_Filter_Trend_Volume
# Hypothesis: Uses 1-day KAMA (Kaufman Adaptive Moving Average) for trend direction,
# combined with 14-period RSI for momentum confirmation, volume surge filter, and 1-week trend filter.
# KAMA adapts to market noise, reducing whipsaws in sideways markets while capturing trends.
# RSI filter ensures entries occur with momentum, avoiding overextended moves.
# Volume surge confirms institutional participation.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for 1d timeframe to target 7-25 trades/year with low frequency and high conviction.
# Works in both bull (KAMA up, RSI > 50) and bear (KAMA down, RSI < 50) markets.

name = "1d_1w_KAMA_Direction_RSI_Filter_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day KAMA for trend direction ---
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start at index 9 for 10-period lookback
    for i in range(10, n):
        if np.isnan(kama[i-1]):
            kama[i] = close[i]
        else:
            kama[i] = kama[i-1] + sc[i-10] * (close[i] - kama[i-1])
    
    # Align KAMA (already 1d, no alignment needed)
    kama_aligned = kama  # same timeframe
    
    # --- 14-period RSI for momentum ---
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    # Pad beginning with NaN
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # --- Volume confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Weekly trend filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for KAMA (10), RSI (14), volume MA (20), weekly EMA (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_aligned[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, volume surge, weekly uptrend
            if (close[i] > kama_aligned[i] and 
                rsi[i] > 50 and 
                volume_surge and 
                ema_50_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, volume surge, weekly downtrend
            elif (close[i] < kama_aligned[i] and 
                  rsi[i] < 50 and 
                  volume_surge and 
                  ema_50_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses below KAMA OR weekly trend turns down
                if (close[i] < kama_aligned[i] or 
                    close[i] < ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA OR weekly trend turns up
                if (close[i] > kama_aligned[i] or 
                    close[i] > ema_50_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals