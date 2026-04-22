#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: Daily Donchian breakout with weekly trend filter and volume spike
    # Uses weekly EMA100 for trend filter, daily Donchian(20) for breakout levels
    # Volume surge confirms breakout strength. Works in bull/bear by filtering with weekly trend.
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA100 trend filter
    ema_1w_100 = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_1w_100_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_100)
    
    # Load daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily levels to higher frequency
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_100_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and weekly uptrend
            if close[i] > high_20_aligned[i] and vol_spike[i] and close[i] > ema_1w_100_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and weekly downtrend
            elif close[i] < low_20_aligned[i] and vol_spike[i] and close[i] < ema_1w_100_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level
            if position == 1:
                if close[i] < low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA100_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0