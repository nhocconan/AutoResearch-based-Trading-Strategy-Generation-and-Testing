#!/usr/bin/env python3
"""
6h 1-Day RSI Extreme with Weekly Volume Regime
Hypothesis: Daily RSI extremes (<30 or >70) combined with weekly volume regime
(high volume = institutional interest) provide high-probability mean reversion
setups. Works in both bull and bear markets by fading extremes when volume
confirms institutional participation, avoiding low-volume traps.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate RSI with proper handling"""
    if len(close) < period + 1:
        return np.full_like(close, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate daily RSI
    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    
    # Get weekly data for volume regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week average volume
    vol_20w_mean = np.zeros(len(df_1w))
    for i in range(len(df_1w)):
        if i < 20:
            vol_20w_mean[i] = np.mean(df_1w['volume'].values[max(0, i-19):i+1])
        else:
            vol_20w_mean[i] = np.mean(df_1w['volume'].values[i-19:i+1])
    
    # Current week volume > 20-week average = high volume regime
    vol_regime = df_1w['volume'].values > vol_20w_mean
    
    # Align to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1w, vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: RSI oversold (<30) in high volume regime
            if (rsi_1d_aligned[i] < 30 and vol_regime_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short setup: RSI overbought (>70) in high volume regime
            elif (rsi_1d_aligned[i] > 70 and vol_regime_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or volume regime ends
            if rsi_1d_aligned[i] > 50 or vol_regime_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or volume regime ends
            if rsi_1d_aligned[i] < 50 or vol_regime_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1D_RSI_Extreme_WeeklyVolumeRegime"
timeframe = "6h"
leverage = 1.0