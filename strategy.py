#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume spike and weekly pivot direction filter
# Long when price breaks above 20-period 6h Donchian high AND 1d volume > 1.8x 20-period volume SMA AND price > weekly pivot
# Short when price breaks below 20-period 6h Donchian low AND same filters
# Donchian provides structure, volume confirms conviction, weekly pivot ensures alignment with higher timeframe bias
# Position size 0.25 to limit drawdown. Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 6h data once before loop for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data once before loop for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 6h Indicator: Donchian Channel (20-period) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Rolling max/min for Donchian channels
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # === 1d Indicator: Volume SMA (20-period) for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1w Indicator: Weekly Pivot (standard calculation) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (need 20 for Donchian and volume SMA)
    warmup = 40
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_series = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_series)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.8x 20-period 1d volume SMA
        vol_threshold = vol_sma_20_1d_aligned[i] * 1.8
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Weekly pivot filter: price above/below weekly pivot for long/short bias
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # === LONG CONDITIONS ===
        # Price breaks above 6h Donchian high AND volume confirmation AND price above weekly pivot
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and price_above_pivot:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Price breaks below 6h Donchian low AND volume confirmation AND price below weekly pivot
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and price_below_pivot:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_1dVolumeSpike_1wPivot_Filter_v1"
timeframe = "6h"
leverage = 1.0