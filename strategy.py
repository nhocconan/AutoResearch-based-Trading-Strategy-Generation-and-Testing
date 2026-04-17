#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d EMA50 trend filter.
Long when price breaks above upper Donchian channel AND volume > 1.5x average AND price > 1d EMA50.
Short when price breaks below lower Donchian channel AND volume > 1.5x average AND price < 1d EMA50.
Exit when price crosses the middle Donchian (20-period SMA of high/low) or weekly trend reverses.
Uses 1d for EMA50 trend filter, 4h for Donchian and volume.
Target: 75-200 total trades over 4 years (19-50/year). Donchian breakouts capture strong moves,
volume confirmation filters weak breakouts, 1d EMA50 ensures alignment with higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels on 4h
    # Upper band: 20-period high
    # Lower band: 20-period low
    # Middle band: 20-period SMA of (high+low)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = ((high_series + low_series) / 2).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        ema50 = ema50_1d_aligned[i]
        vol_ma = volume_ma[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume confirmed AND price > 1d EMA50 (uptrend)
            if price > upper and volume_confirmed and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND volume confirmed AND price < 1d EMA50 (downtrend)
            elif price < lower and volume_confirmed and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle Donchian OR weekly trend reverses (price < 1d EMA50)
            if price < middle or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle Donchian OR weekly trend reverses (price > 1d EMA50)
            if price > middle or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0