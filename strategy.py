#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Donchian breakouts capture strong momentum moves; 1d EMA50 ensures alignment with daily trend
# Volume spike confirms institutional participation. Works in both bull and bear markets by
# only taking breakouts in the direction of the higher timeframe trend.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_v1"
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
    
    # Load HTF data ONCE before loop for 1d EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 (needs 50 periods)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for 1d EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50 = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band + above 1d EMA50 + volume spike
            if (curr_high > curr_highest_high and 
                curr_close > curr_ema_50 and 
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + below 1d EMA50 + volume spike
            elif (curr_low < curr_lowest_low and 
                  curr_close < curr_ema_50 and 
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when price breaks below Donchian lower band
            if curr_low < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price breaks above Donchian upper band
            if curr_high > curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals