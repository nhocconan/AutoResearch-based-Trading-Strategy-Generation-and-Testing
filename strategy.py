#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and weekly EMA trend filter.
# Weekly Donchian channels (20-period) capture major trend breakouts.
# Volume spike (>1.5x 20-period weekly average) confirms institutional participation.
# Weekly EMA50 ensures trades align with higher timeframe trend.
# Designed for very low trade frequency (target: 30-100 total trades over 4 years) to minimize fee drag.
# Works in bull markets (breakout continuations) and bear markets (breakdown continuations).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_max = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (wait for weekly bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_max)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_min)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly volume filter: volume > 1.5x 20-period weekly average
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    volume_filter = volume_1w > (vol_ma_1w_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long breakout: price breaks above weekly Donchian high with uptrend and volume
        if (close[i] > donchian_high_aligned[i] and 
            close[i] > ema50_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short breakdown: price breaks below weekly Donchian low with downtrend and volume
        elif (close[i] < donchian_low_aligned[i] and 
              close[i] < ema50_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions
        elif position == 1 and close[i] <= ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] >= ema50_1w_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0