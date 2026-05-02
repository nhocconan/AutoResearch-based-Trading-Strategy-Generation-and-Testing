#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout for direction and 1d EMA(34) for primary trend filter
# Entry timing on 1h with volume spike confirmation (1.5x 20-period average) to reduce false breakouts
# Uses discrete position sizing 0.20 to minimize fee churn
# Targets 15-25 trades/year (60-100 total over 4 years) to stay within fee drag limits for 1h timeframe
# Works in both bull and bear markets by requiring volume confirmation and primary trend alignment from higher timeframes

name = "1h_Donchian20_4hDirection_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian channels (signal direction)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) - using 4h high/low
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume spike (1.5x 20-period average) on 1h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 55  # max(20 for Donchian/volume, 34 for EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 4h Donchian upper channel + price > 1d EMA + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower channel + price < 1d EMA + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retreats to midpoint of 4h Donchian channel
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises to midpoint of 4h Donchian channel
            midpoint = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals