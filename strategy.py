#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Weekly Mean Reversion using 1w Bollinger Bands and 1d Volume Spike.
# Long when price touches lower 1w Bollinger Band (2 std) with >2x average 1d volume,
# short when price touches upper 1w Bollinger Band (2 std) with >2x average 1d volume.
# Uses mean reversion at extreme weekly levels with volume confirmation to avoid false signals.
# Designed for 15-25 trades/year on 12h timeframe, effective in both bull and bear markets
# as extremes often precede reversals regardless of trend direction.

name = "12h_1w_1d_bollinger_volume_meanrev_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1w Bollinger Bands (20-period, 2 std)
    close_1w = df_1w['close'].values
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    
    # Align 1w Bollinger Bands to 12h timeframe
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    sma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_20_1w)
    
    # Calculate 1d average volume (20-period)
    volume_1d = df_1d['volume'].values
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after Bollinger Band period
        # Skip if any required data is invalid
        if (np.isnan(upper_bb_1w_aligned[i]) or np.isnan(lower_bb_1w_aligned[i]) or 
            np.isnan(sma_20_1w_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume spike condition: current 1d volume > 2x average 1d volume
        volume_spike = volume[i] > 2 * avg_vol_20_1d_aligned[i]
        
        # Entry conditions: price at Bollinger Bands with volume spike
        bb_long_signal = (low[i] <= lower_bb_1w_aligned[i]) and volume_spike
        bb_short_signal = (high[i] >= upper_bb_1w_aligned[i]) and volume_spike
        
        # Exit conditions: price returns to middle of Bollinger Bands
        exit_long = close[i] >= sma_20_1w_aligned[i]
        exit_short = close[i] <= sma_20_1w_aligned[i]
        
        # Priority: entry > exit > hold
        if bb_long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif bb_short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals