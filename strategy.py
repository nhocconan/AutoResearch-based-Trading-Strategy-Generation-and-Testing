#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot (R4/S4) continuation filter and volume confirmation
# - Long when price breaks above Donchian(20) high on 6h AND above 1d weekly pivot R4 with volume spike
# - Short when price breaks below Donchian(20) low on 6h AND below 1d weekly pivot S4 with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Weekly pivot from 1d provides structural levels that work in both bull and bear markets

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(5) for weekly pivot calculation (using daily range)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_5_1d = np.zeros_like(tr)
    for i in range(5, len(tr)):
        atr_5_1d[i] = (atr_5_1d[i-1] * 4 + tr[i]) / 5  # Wilder's smoothing
    
    # 1d weekly pivot points (based on prior week)
    # Need to group by week - use rolling window of 5 trading days approx
    # Weekly high = max(high_1d over 5d), weekly low = min(low_1d over 5d), weekly close = close_1d
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot: P = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R4 = P + 3*(weekly_high - weekly_low)
    # Weekly S4 = P - 3*(weekly_high - weekly_low)
    weekly_range = weekly_high - weekly_low
    weekly_r4 = weekly_pivot + 3.0 * weekly_range
    weekly_s4 = weekly_pivot - 3.0 * weekly_range
    
    # Align weekly pivot levels to 6h
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # 1d volume confirmation: > 1.8x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 6h Donchian(20) breakout levels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or reverts to weekly pivot
            if close_6h[i] < donchian_low[i] or close_6h[i] < weekly_pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or reverts to weekly pivot
            if close_6h[i] > donchian_high[i] or close_6h[i] > weekly_pivot[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with weekly pivot continuation and volume
            if vol_spike_1d_aligned[i]:
                # Long signal: break above Donchian high AND above weekly R4
                if close_6h[i] > donchian_high[i] and close_6h[i] > weekly_r4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short signal: break below Donchian low AND below weekly S4
                elif close_6h[i] < donchian_low[i] and close_6h[i] < weekly_s4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals