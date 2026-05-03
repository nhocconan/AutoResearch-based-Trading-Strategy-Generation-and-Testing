#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian(20) provides robust price channel structure that works in both bull and bear markets
# In bull markets: buy when price breaks above 20-period high with volume spike + price above 1d EMA50
# In bear markets: sell when price breaks below 20-period low with volume spike + price below 1d EMA50
# 1d EMA50 provides long-term trend filter that reduces false breakouts in choppy markets
# Volume confirmation ensures breakouts have conviction
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# This focuses on BTC/ETH by requiring alignment with 1d trend, reducing SOL-only bias

name = "12h_Donchian20_1dEMA50_VolumeSpike"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 12h timeframe
    lookback = 20
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute rolling high/low for efficiency
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    rolling_high = high_series.rolling(window=lookback, min_periods=lookback).max().values
    rolling_low = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    for i in range(lookback, n):  # Start from lookback to have valid Donchian channels
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rolling_high[i]) or 
            np.isnan(rolling_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Donchian breakout signals with 1d trend filter
        # Long: price breaks above 20-period high + volume spike + price above 1d EMA50
        # Short: price breaks below 20-period low + volume spike + price below 1d EMA50
        if position == 0:
            if (close[i] > rolling_high[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < rolling_low[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-period low (reversal) OR price below 1d EMA50
            if close[i] < rolling_low[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-period high (reversal) OR price above 1d EMA50
            if close[i] > rolling_high[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals