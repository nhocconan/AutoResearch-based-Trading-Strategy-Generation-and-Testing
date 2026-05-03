#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze + 1d Volume Spike Regime.
# In low volatility regimes (BB Width < 20th percentile on 6h), wait for 1d volume spike (>2x 20-day average) 
# to trigger breakout trades in direction of 1d close vs open. Uses Bollinger Bands (20,2) and volume confirmation
# to capture explosive moves after consolidation, effective in both bull and bear markets by trading breakouts
# from squeezes with institutional volume validation. Target: 12-37 trades/year.

name = "6h_BBSqueeze_1dVolSpike_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_time = prices['open_time']
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d average volume (20-period) for spike detection
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (2.0 * avg_vol_1d)  # Volume > 2x 20-day average
    
    # Align 1d volume spike to 6h timeframe
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 6h Bollinger Bands (20,2) for squeeze detection
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2.0 * std_20)
    lower_bb = sma_20 - (2.0 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate 20th percentile of BB Width for squeeze threshold (using expanding window)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(20, len(bb_width)):
        bb_width_percentile[i] = np.percentile(bb_width[20:i+1], 20)
    
    is_squeeze = bb_width < bb_width_percentile  # BB Width below 20th percentile
    
    # Breakout direction: 1d close vs open (bullish if close > open)
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    bullish_1d = close_1d > open_1d
    bullish_1d_aligned = align_htf_to_ltf(prices, df_1d, bullish_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(bullish_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Entry conditions: Bollinger squeeze + 1d volume spike + directional bias
        is_squeeze_now = is_squeeze[i]
        has_vol_spike = vol_spike_1d_aligned[i] > 0.5
        is_bullish_bias = bullish_1d_aligned[i] > 0.5
        
        if position == 0:
            # Long: squeeze + volume spike + bullish 1d bias
            if is_squeeze_now and has_vol_spike and is_bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: squeeze + volume spike + bearish 1d bias
            elif is_squeeze_now and has_vol_spike and not is_bullish_bias:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: squeeze ends OR reverse volume spike with bearish bias
            if not is_squeeze_now or (has_vol_spike and not is_bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: squeeze ends OR reverse volume spike with bullish bias
            if not is_squeeze_now or (has_vol_spike and is_bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals