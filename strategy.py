#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakouts with volume confirmation
# Price breaking above/below 4h Donchian channel (20-period high/low) indicates trend continuation
# Volume > 1.5x 20-period average confirms institutional participation
# Use 1d EMA50 as trend filter: only long when price > EMA50, short when price < EMA50
# Trade only during active session (08-20 UTC) to reduce noise
# Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing
# Works in both bull/bear markets: breakouts capture trends, EMA filter avoids counter-trend trades

name = "1h_Donchian20_4hBreakout_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period high/low) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian channel: 20-period high and low
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_low = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Calculate 1d EMA50 for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + volume + above 1d EMA50
            if close[i] > donchian_high[i] and volume_filter[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low + volume + below 1d EMA50
            elif close[i] < donchian_low[i] and volume_filter[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low (trend failure)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high (trend failure)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals