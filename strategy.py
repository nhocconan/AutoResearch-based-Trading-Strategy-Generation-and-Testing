#!/usr/bin/env python3
"""
6h_volume_profile_candle_clustering_v1
Hypothesis: On 6-hour timeframe, identify high-probability breakout points using volume profile clustering.
Long when price breaks above the 70th percentile volume node with increasing volume.
Short when price breaks below the 30th percentile volume node with increasing volume.
Volume profile clusters act as institutional support/resistance where large traders accumulate/distribute.
This strategy works in both bull/bear markets as volume clusters adapt to price action and breakouts
from high-volume areas indicate strong institutional participation.
Target: 15-35 trades/year to minimize fee impact while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volume_profile_candle_clustering_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume profile
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate volume profile on daily timeframe (volume-weighted price clusters)
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Create price bins for volume profile (20 bins between min and max of last 60 days)
    lookback = min(60, len(df_1d))
    price_min = np.min(np.concatenate([low[-lookback*4:], df_1d['low'].values[-lookback:]])) if lookback*4 <= len(low) else np.min(low[-lookback*4:])
    price_max = np.max(np.concatenate([high[-lookback*4:], df_1d['high'].values[-lookback:]])) if lookback*4 <= len(high) else np.max(high[-lookback*4:])
    
    if price_max <= price_min:
        price_max = price_min + 1.0
    
    bins = 20
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Initialize volume profile array
    volume_profile = np.zeros(bins)
    
    # Calculate volume profile for the lookback period
    start_idx = max(0, len(df_1d) - lookback)
    for i in range(start_idx, len(df_1d)):
        price = (df_1d['high'].iloc[i] + df_1d['low'].iloc[i] + df_1d['close'].iloc[i]) / 3
        vol = df_1d['volume'].iloc[i]
        # Find which bin this price falls into
        bin_idx = np.digitize(price, bin_edges) - 1
        if 0 <= bin_idx < bins:
            volume_profile[bin_idx] += vol
    
    # Find high volume nodes (70th percentile and above, 30th percentile and below)
    sorted_indices = np.argsort(volume_profile)[::-1]
    total_volume = np.sum(volume_profile)
    
    # Calculate cumulative volume to find percentiles
    cum_volume = 0
    vol_70_idx = bins - 1
    vol_30_idx = 0
    
    for idx in sorted_indices:
        cum_volume += volume_profile[idx]
        if cum_volume >= 0.7 * total_volume:
            vol_70_idx = idx
            break
    
    cum_volume = 0
    for idx in sorted_indices:
        cum_volume += volume_profile[idx]
        if cum_volume >= 0.3 * total_volume:
            vol_30_idx = idx
            break
    
    # Get price levels for high volume nodes
    high_vol_node_70 = bin_centers[vol_70_idx] if vol_70_idx < len(bin_centers) else price_max
    high_vol_node_30 = bin_centers[vol_30_idx] if vol_30_idx < len(bin_centers) else price_min
    
    # Smooth the volume profile levels to avoid noise
    vol_node_70_series = pd.Series([high_vol_node_70] * len(df_1d)).rolling(window=5, center=True, min_periods=1).mean().values
    vol_node_30_series = pd.Series([high_vol_node_30] * len(df_1d)).rolling(window=5, center=True, min_periods=1).mean().values
    
    # Align volume profile levels to 6h timeframe
    vol_node_70_aligned = align_htf_to_ltf(prices, df_1d, vol_node_70_series)
    vol_node_30_aligned = align_htf_to_ltf(prices, df_1d, vol_node_30_series)
    
    # Volume confirmation: 20-period volume average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(30, 20), n):
        # Skip if data not available
        if (np.isnan(vol_node_70_aligned[i]) or np.isnan(vol_node_30_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.3 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes back below the 70th percentile volume node
            if close[i] < vol_node_70_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back above the 30th percentile volume node
            if close[i] > vol_node_30_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price closes above 70th percentile volume node with increasing volume
                if (close[i] > vol_node_70_aligned[i] and close[i-1] <= vol_node_70_aligned[i-1] and 
                    volume[i] > volume[i-1]):
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below 30th percentile volume node with increasing volume
                elif (close[i] < vol_node_30_aligned[i] and close[i-1] >= vol_node_30_aligned[i-1] and 
                      volume[i] > volume[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals