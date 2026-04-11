#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return signals
    
    # Calculate daily pivot point (previous day's close for causality)
    close_1d = df_1d['close'].values
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    
    # Use previous day's close as anchor for today's bias (no look-ahead)
    bias_up = close_1d > prev_close_1d  # today's close > yesterday's close
    bias_down = close_1d < prev_close_1d  # today's close < yesterday's close
    
    # Align bias to 4h timeframe (available after daily bar closes)
    bias_up_aligned = align_htf_to_ltf(prices, df_1d, bias_up)
    bias_down_aligned = align_htf_to_ltf(prices, df_1d, bias_down)
    
    # Volume confirmation: volume > 1.5x 20-period average (more selective)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bias_up_aligned[i]) or np.isnan(bias_down_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        bias_up = bias_up_aligned[i]
        bias_down = bias_down_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.5 * vol_ma_20[i]
        
        # Trading logic based on daily bias + volume
        long_signal = bias_up and volume_confirmed
        short_signal = bias_down and volume_confirmed
        
        # Exit on bias reversal or volume fade
        exit_long = bias_down or not volume_confirmed
        exit_short = bias_up or not volume_confirmed
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 4h strategy using daily price bias (close > yesterday close = up bias) 
# with volume confirmation. Enters long on up bias + volume, short on down bias + volume. 
# Exits when bias reverses or volume fades. Works in both bull and bear markets 
# by following the daily momentum filtered by volume. Target: 20-40 trades/year 
# to minimize fee decay while capturing sustained moves. Uses 4h timeframe for 
# execution but relies on 1d bias for direction to avoid overtrading. Volume 
# filter (>1.5x 20-period average) ensures institutional participation. 
# Simple, robust, and avoids overtrading pitfalls of similar strategies.