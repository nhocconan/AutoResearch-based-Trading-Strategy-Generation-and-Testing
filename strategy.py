#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume spike confirmation.
# Works in bull/bear by using 1d EMA50 trend filter + volume spike for momentum confirmation.
# Target: 20-50 trades/year (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Load 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower (20-period)
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, high_20_4h)
    low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, low_20_4h)
    
    # Calculate 4h volume average (20-period)
    volume_4h = df_4h['volume'].values
    vol_avg_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20_4h_aligned[i]) or 
            np.isnan(low_20_4h_aligned[i]) or np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper in uptrend with volume spike
            if (close[i] > high_20_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20_4h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower in downtrend with volume spike
            elif (close[i] < low_20_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20_4h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0