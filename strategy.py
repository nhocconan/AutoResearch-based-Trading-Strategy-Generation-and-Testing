#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot (R4/S4) continuation filter and volume confirmation
# - Long when price breaks above 6h Donchian upper (20) AND 1d price > weekly R4 pivot with volume spike
# - Short when price breaks below 6h Donchian lower (20) AND 1d price < weekly S4 pivot with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly pivot calculated from prior week: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d weekly pivot levels (based on prior week: Mon-Fri)
    # Calculate weekly OHLC from daily data
    # We'll approximate: weekly high = max(high_1d over 5 days), etc.
    # But simpler: use prior 5-day range for weekly pivot
    # R4 = close + 1.5*(weekly_high - weekly_low)
    # S4 = close - 1.5*(weekly_high - weekly_low)
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_range = weekly_high - weekly_low
    weekly_r4 = close_1d + 1.5 * weekly_range
    weekly_s4 = close_1d - 1.5 * weekly_range
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1d, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1d, weekly_s4)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
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
            # Exit: price breaks below Donchian lower or loses weekly R4 support
            if (prices['close'].iloc[i] < donchian_low[i] or 
                prices['close'].iloc[i] < weekly_r4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian higher or loses weekly S4 resistance
            if (prices['close'].iloc[i] > donchian_high[i] or 
                prices['close'].iloc[i] > weekly_s4_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with weekly pivot and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above Donchian higher AND above weekly R4
                if (prices['close'].iloc[i] > donchian_high[i] and 
                    prices['close'].iloc[i] > weekly_r4_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian lower AND below weekly S4
                elif (prices['close'].iloc[i] < donchian_low[i] and 
                      prices['close'].iloc[i] < weekly_s4_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals