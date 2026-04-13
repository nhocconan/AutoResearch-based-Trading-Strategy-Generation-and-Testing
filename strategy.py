#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Bands squeeze and mean reversion.
# Long: Bollinger Bands width at 20-day low + price below lower band + volume spike.
# Short: Bollinger Bands width at 20-day low + price above upper band + volume spike.
# Exit: Price crosses back inside Bollinger Bands.
# Bollinger squeeze indicates low volatility, often preceding mean-reverting moves.
# Volume spike confirms participation in the breakout/mean reversion.
# Works in both bull and bear markets as it captures reversals from overextended conditions.
# Target: 20-50 trades per year (~80-200 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20-period, 2 std dev)
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    bb_upper = np.full(len(close_1d), np.nan)
    bb_lower = np.full(len(close_1d), np.nan)
    bb_width = np.full(len(close_1d), np.nan)
    
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
        bb_upper[i] = sma_20[i] + 2 * std_20[i]
        bb_lower[i] = sma_20[i] - 2 * std_20[i]
        bb_width[i] = bb_upper[i] - bb_lower[i]
    
    # Bollinger Bands width percentile (20-day lookback)
    bb_width_percentile = np.full(len(bb_width), np.nan)
    for i in range(40, len(bb_width)):  # Need 20 for BB + 20 for percentile lookback
        window = bb_width[i-20:i]
        if not np.all(np.isnan(window)):
            bb_width_percentile[i] = (np.sum(window < bb_width[i]) / 20) * 100
    
    # Average volume (20-period) for volume confirmation
    avg_volume_1d = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        avg_volume_1d[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(40, n):  # Start after sufficient lookback
        # Skip if any required data is not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume_1d_aligned[i]
        upper = bb_upper_aligned[i]
        lower = bb_lower_aligned[i]
        width_percentile = bb_width_percentile_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        # Bollinger squeeze condition: width at or below 10th percentile (low volatility)
        squeeze_condition = width_percentile <= 10
        
        if position == 0:
            # Long: squeeze + price at or below lower band + volume spike
            if (squeeze_condition and 
                price <= lower and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: squeeze + price at or above upper band + volume spike
            elif (squeeze_condition and 
                  price >= upper and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above the middle (SMA) or stop if too adverse
            if price >= sma_20[i] if i < len(sma_20) and not np.isnan(sma_20[i]) else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below the middle (SMA)
            if price <= sma_20[i] if i < len(sma_20) and not np.isnan(sma_20[i]) else False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Bollinger_Squeeze_MeanReversion"
timeframe = "4h"
leverage = 1.0