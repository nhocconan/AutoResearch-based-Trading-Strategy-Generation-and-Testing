#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian channel breakout with 12h trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian upper band with 12h uptrend and volume > 1.5x average.
# Short when price breaks below 20-period Donchian lower band with 12h downtrend and volume > 1.5x average.
# Uses 6h for entry timing, 12h for trend filter to reduce whipsaw. Designed for ~20-30 trades/year per symbol.
# Works in both bull and bear markets by following 12h trend and requiring volatility expansion via volume.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Calculate Donchian channels (20-period) on 6h
    upper_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    lower_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_6h_aligned = align_htf_to_ltf(prices, df_6h, upper_6h)
    lower_6h_aligned = align_htf_to_ltf(prices, df_6h, lower_6h)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 50-period EMA on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: volume > 1.5x 24-period average (4 hours worth of 6h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_6h_aligned[i]) or np.isnan(lower_6h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: break above upper band AND 12h uptrend AND volume filter
        if (close[i] > upper_6h_aligned[i] and 
            close[i] > ema50_12h_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: break below lower band AND 12h downtrend AND volume filter
        elif (close[i] < lower_6h_aligned[i] and 
              close[i] < ema50_12h_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_DonchianBreakout_12hEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0