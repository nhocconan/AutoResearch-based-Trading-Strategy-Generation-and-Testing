#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily KAMA trend with weekly RSI filter and volume confirmation.
# Uses KAMA(10) for trend direction on daily timeframe, filtered by weekly RSI(14) > 50 for longs and < 50 for shorts.
# Volume confirmation requires current volume > 1.5x 20-day average volume.
# Designed for low-frequency, high-conviction trades (target: 10-25 trades/year) to minimize fee drag.
# Works in both bull and bear markets by adapting trend direction via KAMA and filtering extremes with weekly RSI.

name = "1d_kama_weekly_rsi_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, 10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d, 1)), axis=0)  # 10-period volatility
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+2) - 2/(30+2)) + 2/(30+2)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]  # seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i-10] * (close_1d[i] - kama[i-1])
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average
    avg_gain = np.full_like(close_1w, np.nan, dtype=float)
    avg_loss = np.full_like(close_1w, np.nan, dtype=float)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    # Wilder smoothing
    for i in range(15, len(close_1w)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan, dtype=float), where=avg_loss!=0)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Calculate 20-day average volume on daily
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align indicators to daily timeframe (no shift needed as daily is base)
    kama_aligned = kama  # already daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-day average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # KAMA trend: price above KAMA = bullish, below = bearish
        is_bullish = close[i] > kama_aligned[i]
        is_bearish = close[i] < kama_aligned[i]
        
        # Weekly RSI filter: RSI > 50 for long bias, < 50 for short bias
        rsi_long_filter = rsi_1w_aligned[i] > 50
        rsi_short_filter = rsi_1w_aligned[i] < 50
        
        # Enter long: price above KAMA, volume confirmation, weekly RSI > 50
        enter_long = is_bullish and vol_filter and rsi_long_filter
        # Enter short: price below KAMA, volume confirmation, weekly RSI < 50
        enter_short = is_bearish and vol_filter and rsi_short_filter
        
        # Exit when price crosses KAMA in opposite direction
        exit_long = position == 1 and close[i] < kama_aligned[i]
        exit_short = position == -1 and close[i] > kama_aligned[i]
        
        # Update position and signal
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals