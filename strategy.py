#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with weekly ATR filter and volume confirmation.
Buy when price breaks above 20-day high with above-average volume and low weekly volatility.
Sell when price breaks below 20-day low with above-average volume and low weekly volatility.
Weekly ATR filter avoids choppy markets. Designed for 10-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mts_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate Average True Range."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    
    atr = np.full(len(tr), np.nan)
    atr[period] = np.nanmean(tr[:period+1])
    
    for i in range(period + 1, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data for ATR filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = np.full(len(high_1d), np.nan)
    low_20 = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 20-period ATR on weekly
    atr_20_1w = calculate_atr(high_1w, low_1w, close_1w, 20)
    
    # Calculate 20-period average volume on daily
    vol_avg_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_avg_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align to daily timeframe (already daily, but for consistency)
    high_20_aligned = high_20  # already aligned to daily
    low_20_aligned = low_20
    atr_20_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_20_1w)
    vol_avg_20_1d_aligned = vol_avg_20_1d  # already aligned to daily
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(atr_20_1w_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_confirmed = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Weekly ATR filter: avoid extremely high volatility (above 80th percentile)
        # We'll use a simple threshold: ATR < 0.03 * price (3% of price)
        atr_filter = atr_20_1w_aligned[i] < 0.03 * close[i]
        
        if position == 0:
            # Long: price breaks above 20-day high, volume confirmation, low volatility
            if close[i] > high_20_aligned[i] and vol_confirmed and atr_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low, volume confirmation, low volatility
            elif close[i] < low_20_aligned[i] and vol_confirmed and atr_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day low or volatility spikes
            if close[i] < low_20_aligned[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day high or volatility spikes
            if close[i] > high_20_aligned[i] or not atr_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "Daily_Donchian20_WeeklyATR_Volume"
timeframe = "1d"
leverage = 1.0