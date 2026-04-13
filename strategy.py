#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Bollinger Band squeeze breakout + volume confirmation.
# Long: Price breaks above upper Bollinger Band (20,2) + volume > 1.5x average volume.
# Short: Price breaks below lower Bollinger Band (20,2) + volume > 1.5x average volume.
# Uses Bollinger Band squeeze (bandwidth < 20th percentile) as volatility contraction filter.
# Bollinger Bands provide dynamic support/resistance that adapts to volatility.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20,2) on daily closes
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    bandwidth = (upper_band - lower_band) / sma_20  # Normalized bandwidth
    
    # Calculate 20th percentile of bandwidth for squeeze condition
    bandwidth_percentile_20 = np.full(len(bandwidth), np.nan)
    for i in range(50, len(bandwidth)):  # Need enough data for percentile
        bandwidth_percentile_20[i] = np.percentile(bandwidth[max(0, i-50):i+1], 20)
    
    # Squeeze condition: bandwidth < 20th percentile (low volatility)
    squeeze_condition = bandwidth < bandwidth_percentile_20
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Align 1d Bollinger Bands and squeeze to 12h
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze_condition)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        squeeze = squeeze_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above upper band + squeeze + volume confirmation
            if (price > upper and squeeze and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower band + squeeze + volume confirmation
            elif (price < lower and squeeze and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below middle Bollinger Band (SMA20)
            middle_band = sma_20[np.searchsorted(df_1d.index, prices.index[i])] if i < len(prices) else sma_20[-1]
            # Simplified exit: price < midpoint of bands
            midpoint = (upper + lower) / 2
            if price < midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above middle Bollinger Band
            midpoint = (upper + lower) / 2
            if price > midpoint:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Bollinger_Squeeze_Breakout_Volume"
timeframe = "12h"
leverage = 1.0