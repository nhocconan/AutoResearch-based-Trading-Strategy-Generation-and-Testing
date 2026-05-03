#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Donchian breakouts capture strong momentum moves; 1w EMA50 ensures alignment with weekly trend
# Volume spike (>1.8x 20-period EMA) confirms breakout validity
# Target: 8-18 trades/year (30-70 total over 4 years) to minimize fee drag
# Works in bull/bear: trend filter prevents counter-trend entries during reversals

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels on 1d data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Donchian breakout signals with 1w trend filter
        # Long: price breaks above Donchian upper + price above 1w EMA50 + volume spike
        # Short: price breaks below Donchian lower + price below 1w EMA50 + volume spike
        if position == 0:
            if (close[i] > highest_high[i-1] and 
                close[i] > ema_50_1w_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            elif (close[i] < lowest_low[i-1] and 
                  close[i] < ema_50_1w_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle OR below 1w EMA50
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] < donchian_middle or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian middle OR above 1w EMA50
            donchian_middle = (highest_high[i] + lowest_low[i]) / 2.0
            if close[i] > donchian_middle or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals