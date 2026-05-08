#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter and volume comparison
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume average for volume spike detection
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate Donchian channels on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 12h volume > 1.5x daily average
        volume_spike = volume[i] > (1.5 * vol_avg_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Donchian high + daily uptrend + volume spike
            long_cond = (close[i] > highest_high[i]) and (close[i] > ema34_1d_aligned[i]) and volume_spike
            
            # Short: price breaks below Donchian low + daily downtrend + volume spike
            short_cond = (close[i] < lowest_low[i]) and (close[i] < ema34_1d_aligned[i]) and volume_spike
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loses daily uptrend
            if (close[i] < lowest_low[i]) or (close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loses daily downtrend
            if (close[i] > highest_high[i]) or (close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian breakouts capture strong trends, filtered by daily EMA34 trend and volume spikes.
# Long when price breaks above 20-period high, above daily EMA34 (uptrend), with volume confirmation.
# Short when price breaks below 20-period low, below daily EMA34 (downtrend), with volume confirmation.
# Exits when price breaks opposite Donchian band or loses daily trend alignment.
# Works in bull markets (catching breakouts) and bear markets (shorting breakdowns).
# Volume spike ensures participation, reducing false breakouts.
# Target: 50-150 total trades over 4 years = 12-37/year to minimize fee decay.