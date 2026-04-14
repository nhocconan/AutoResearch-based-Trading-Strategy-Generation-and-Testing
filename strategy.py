#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using 1-week Bollinger Bands with 1-day volume confirmation.
# Long when price touches lower BB on 1w with high volume (mean reversion).
# Short when price touches upper BB on 1w with high volume (mean reversion).
# Exit when price returns to 1w middle band (SMA).
# Uses 1-week timeframe for structural levels and 1-day for volume confirmation to reduce noise.
# Designed to work in both bull and bear markets by capturing mean reversion at extremes.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for structural levels (Bollinger Bands)
    df_1w = get_htf_data(prices, '1w')
    
    # Load 1d data ONCE for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1w Bollinger Bands (20, 2)
    bb_len = 20
    bb_std = 2
    if len(df_1w) < bb_len:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_1w = pd.Series(close_1w).rolling(window=bb_len, min_periods=bb_len).mean().values
    std_1w = pd.Series(close_1w).rolling(window=bb_len, min_periods=bb_len).std().values
    bb_upper_1w = sma_1w + bb_std * std_1w
    bb_lower_1w = sma_1w - bb_std * std_1w
    
    # Align 1w Bollinger Bands to 12h timeframe
    bb_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_upper_1w)
    bb_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, bb_lower_1w)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # 1-day volume confirmation (20-period average)
    vol_ma_len = 20
    if len(df_1d) < vol_ma_len:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=vol_ma_len, min_periods=vol_ma_len).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(bb_len, vol_ma_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper_1w_aligned[i]) or 
            np.isnan(bb_lower_1w_aligned[i]) or
            np.isnan(sma_1w_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        # Convert 1d volume average to 12h equivalent (approximate)
        volume_confirmed = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: price touches or crosses below lower Bollinger Band with volume confirmation
            if (close[i] <= bb_lower_1w_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price touches or crosses above upper Bollinger Band with volume confirmation
            elif (close[i] >= bb_upper_1w_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to or above the 1w middle band (SMA)
            if close[i] >= sma_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to or below the 1w middle band (SMA)
            if close[i] <= sma_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1wBB_1dVol_MeanRev_v1"
timeframe = "12h"
leverage = 1.0