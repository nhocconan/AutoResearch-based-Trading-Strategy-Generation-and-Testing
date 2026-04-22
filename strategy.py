#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d trend filter and volume confirmation
    # Uses daily EMA50 for trend direction, Donchian breakout for entry, volume spike for confirmation
    # Works in bull/bear: breakouts with momentum capture, trend filter prevents counter-trend trades
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian(20) channels
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above upper Donchian with volume spike and 1d uptrend
            if prices['high'].values[i] > high_20[i] and vol_spike[i] and close_1d[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian with volume spike and 1d downtrend
            elif prices['low'].values[i] < low_20[i] and vol_spike[i] and close_1d[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Return to opposite Donchian level or trend reversal
            if position == 1:
                if prices['low'].values[i] < low_20[i] or close_1d[i] < ema_1d_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if prices['high'].values[i] > high_20[i] or close_1d[i] > ema_1d_50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian_Breakout_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0