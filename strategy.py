#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume confirmation and 1d trend filter
# - Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period average + price > 1d EMA50
# - Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period average + price < 1d EMA50
# - Exit when price crosses 1d EMA50 in opposite direction
# - Designed to capture trend continuation with volume confirmation in both bull and bear markets
# - Target: 20-30 trades/year to avoid excessive fee drift

name = "12h_Donchian20_1dVolume_EMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and trend filters
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d has 2x 12h bars, so divide by 2
        volume_factor = vol_ma_1d_aligned[i] / 2.0
        volume_filter = volume[i] > 1.5 * volume_factor
        
        if position == 0:
            # Look for long entry: price > Donchian high + volume + uptrend
            if close[i] > highest_high[i] and volume_filter and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price < Donchian low + volume + downtrend
            elif close[i] < lowest_low[i] and volume_filter and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below 1d EMA50
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above 1d EMA50
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals