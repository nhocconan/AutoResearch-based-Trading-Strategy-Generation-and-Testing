#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for trend direction and 20-period Donchian channels from 12h data for breakout signals
# Volume confirmation requires 2.0x average volume to ensure strong participation and reduce false breakouts
# Trend filter ensures trades only taken in direction of higher timeframe trend
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
# Works in both bull and bear markets by following the 1d trend direction
# Prioritizes BTC/ETH performance with SOL as secondary

name = "12h_Donchian20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels from 12h data (20-period)
    # Upper channel: highest high of last 20 periods
    # Lower channel: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Donchian breakout with 1d trend filter
        # Long: Price breaks above Donchian upper channel + volume spike + price above 1d EMA50 (uptrend)
        # Short: Price breaks below Donchian lower channel + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > donchian_upper[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < donchian_lower[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian lower channel OR price below 1d EMA50 (trend change)
            if close[i] < donchian_lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian upper channel OR price above 1d EMA50 (trend change)
            if close[i] > donchian_upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals