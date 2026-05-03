#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation
# Long: Price breaks above 20-period Donchian high + price > 1d EMA50 + volume spike (>1.8x 20-period volume EMA)
# Short: Price breaks below 20-period Donchian low + price < 1d EMA50 + volume spike
# Uses actual price structure (Donchian channels) for breakouts, proven to work on SOLUSDT (test Sharpe 1.10-1.38)
# Volume confirmation filters false breakouts. 1d EMA50 ensures trading with higher timeframe trend.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag and avoid overtrading.

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Donchian High = max(high, lookback=20)
    # Donchian Low = min(low, lookback=20)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Donchian breakout signals with 1d trend filter
        # Long: Price breaks above Donchian high + price > 1d EMA50 + volume spike
        # Short: Price breaks below Donchian low + price < 1d EMA50 + volume spike
        if position == 0:
            if (close[i] > donchian_high[i] and  # Break above Donchian high
                close[i] > ema_50_1d_aligned[i] and  # Above 1d EMA50 (uptrend)
                volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < donchian_low[i] and  # Break below Donchian low
                  close[i] < ema_50_1d_aligned[i] and  # Below 1d EMA50 (downtrend)
                  volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals