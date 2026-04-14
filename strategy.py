# USDC/USDT mean reversion with Bollinger Bands on 12h timeframe
# Hypothesis: During periods of low volatility, stablecoin pairs like USDC/USDT tend to mean revert
# to the 1.00 peg. Bollinger Band touches with volume confirmation provide entry signals,
# while Bollinger Band width serves as a volatility filter to avoid trending markets.
# This should work in both bull and bear markets as stablecoin deviations are market-independent.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands (daily)
    if len(close_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period SMA of daily close
    sma20_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        sma20_1d[i] = np.mean(close_1d[i-19:i+1])
    
    # Calculate 20-period standard deviation of daily close
    std20_1d = np.full_like(close_1d, np.nan)
    for i in range(19, len(close_1d)):
        std20_1d[i] = np.std(close_1d[i-19:i+1])
    
    # Bollinger Bands: 2 std dev from SMA
    upper_bb_1d = sma20_1d + 2 * std20_1d
    lower_bb_1d = sma20_1d - 2 * std20_1d
    
    # Align Bollinger Bands to 12h timeframe
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma20_1d)
    
    # Bollinger Band Width as volatility filter (narrow = low vol, good for mean reversion)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Volume moving average (20-period) for volume confirmation
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative size to limit trades
    
    for i in range(20, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(sma20_1d_aligned[i]) or
            np.isnan(bb_width_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volatility filter: only trade when Bollinger Bands are narrow (low volatility)
        # Threshold of 0.02 (2%) corresponds to relatively tight bands
        if bb_width_1d_aligned[i] > 0.02:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price touches lower Bollinger Band with volume surge
            if (close[i] <= lower_bb_1d_aligned[i] and 
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price touches upper Bollinger Band with volume surge
            elif (close[i] >= upper_bb_1d_aligned[i] and 
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle (SMA) or volatility increases
            if (close[i] >= sma20_1d_aligned[i] or
                bb_width_1d_aligned[i] > 0.03):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle (SMA) or volatility increases
            if (close[i] <= sma20_1d_aligned[i] or
                bb_width_1d_aligned[i] > 0.03):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_USDC_MeanReversion_BB"
timeframe = "12h"
leverage = 1.0