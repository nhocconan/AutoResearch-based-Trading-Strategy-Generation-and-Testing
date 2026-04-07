#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with 1d Trend Filter and Volume Confirmation v1
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture significant momentum moves.
# Combined with 1d EMA trend filter and volume spike confirmation to filter false breakouts.
# Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
# Target: 12-37 trades per year (50-150 over 4 years).

name = "12h_donchian20_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate volume average (20-period) on 12h data
    volume_12h = df_12h['volume'].values
    volume_series = pd.Series(volume_12h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_ok = volume[i] > 1.5 * volume_ma_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend turns bearish
            if close[i] < donchian_low_aligned[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend turns bullish
            if close[i] > donchian_high_aligned[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high AND uptrend AND volume confirmation
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_1d_aligned[i] and volume_ok):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND downtrend AND volume confirmation
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_1d_aligned[i] and volume_ok):
                position = -1
                signals[i] = -0.25
    
    return signals