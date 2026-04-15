#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d RSI + Volume Spike + Weekly Trend Filter
# Uses RSI(14) for overbought/oversold conditions on daily timeframe.
# Enters long when RSI < 30 and volume > 2x average volume.
# Enters short when RSI > 70 and volume > 2x average volume.
# Weekly trend filter: only take longs when price > weekly EMA(50), shorts when price < weekly EMA(50).
# Works in both bull and bear markets by fading extremes in the direction of the weekly trend.
# Target: 30-80 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on daily closes
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily RSI to 1d timeframe (already aligned)
    # Align weekly EMA(50) to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(close, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    
    # Start from index where we have sufficient data
    start_idx = 14  # Need at least 14 days for RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1w_aligned[i])):
            continue
        
        # Long entry: RSI < 30 (oversold) + volume spike + price above weekly EMA(50)
        if (rsi[i] < 30 and
            volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
            close[i] > ema_50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: RSI > 70 (overbought) + volume spike + price below weekly EMA(50)
        elif (rsi[i] > 70 and
              volume[i] > 2.0 * np.median(volume[max(0, i-20):i+1]) and
              close[i] < ema_50_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: RSI returns to neutral zone (40-60) or reverse signal
        elif position == 1 and (rsi[i] > 50 or rsi[i] < 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 50 or rsi[i] > 70):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_RSI_Volume_Spike_WeeklyTrend"
timeframe = "1d"
leverage = 1.0