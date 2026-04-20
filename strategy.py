#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA Trend with Weekly RSI Filter and Volume Confirmation
# Uses Kaufman's Adaptive Moving Average (KAMA) to detect trend direction on daily timeframe.
# Enters long when price is above KAMA, weekly RSI > 50, and volume > 1.5x average.
# Enters short when price is below KAMA, weekly RSI < 50, and volume > 1.5x average.
# Exits when price crosses back below/above KAMA.
# KAMA adapts to market noise, reducing whipsaw in choppy markets.
# Weekly RSI ensures alignment with higher timeframe momentum.
# Volume confirms institutional participation.
# Target: 15-25 trades/year.

name = "1d_KAMA_WeeklyRSI_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for RSI
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close = prices['close'].values
    direction = np.abs(np.diff(close, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close, 1)), axis=0)  # 10-period sum of absolute changes
    # Avoid division by zero
    er = np.where(volatility > 0, direction / volatility, 0)
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[:10].mean()  # Start with 10-period SMA
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate weekly RSI (14-period)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    # Align weekly RSI to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation (daily)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Get values
        close_val = close[i]
        kama_val = kama[i]
        rsi_val = rsi_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(rsi_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price > KAMA, RSI > 50, volume spike
            if close_val > kama_val and rsi_val > 50 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short entry: price < KAMA, RSI < 50, volume spike
            elif close_val < kama_val and rsi_val < 50 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA
            if close_val <= kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA
            if close_val >= kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals