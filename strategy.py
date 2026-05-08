#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_RVOL_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and RVOL calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = (close_1d > ema50_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 6h RVOL (relative volume): current volume / 20-period average volume
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    rvol = volume / np.where(vol_ma20 > 0, vol_ma20, 1)  # Avoid division by zero
    
    # Get 6h high/low for breakout levels (20-period Donchian channels)
    high_ma20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_ma20[i]) or np.isnan(low_ma20[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(rvol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-period high with high RVOL and 1d uptrend
            long_cond = (close[i] > high_ma20[i] and rvol[i] > 2.0 and trend_1d_aligned[i] > 0.5)
            
            # Short entry: price breaks below 20-period low with high RVOL and 1d downtrend
            short_cond = (close[i] < low_ma20[i] and rvol[i] > 2.0 and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 20-period low (reversal signal)
            if close[i] < low_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 20-period high (reversal signal)
            if close[i] > high_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h RVOL breakout strategy with 1d trend filter. 
# Uses relative volume (RVOL > 2.0) to identify high-momentum breakouts from 20-period Donchian channels.
# 1d EMA50 provides trend filter to avoid counter-trend trades.
# Exit on reversal: close breaks opposite Donchian boundary.
# Targets 15-25 trades/year to minimize fee drag while capturing explosive moves in both bull and bear markets.
# RVOL filter ensures trades occur only during institutional participation, reducing whipsaws.