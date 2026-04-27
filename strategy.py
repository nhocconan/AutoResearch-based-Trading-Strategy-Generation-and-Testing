#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6H Donchian(20) breakout with 1D EMA50 trend filter and volume spike.
# Uses daily EMA50 to determine trend direction (bull/bear) and enters on breakout
# of 6H Donchian channels with volume confirmation. In strong trends (price > EMA50),
# long breakouts are favored; in weak trends (price < EMA50), short breakdowns are favored.
# Volume spike confirms institutional participation. Designed for ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1D data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 50-period EMA on 1D close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get daily high/low for context (not used in entry, but could be used for filtering)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    
    # Volume filter: volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        
        # Long entry: breakout above Donchian high in uptrend with volume
        if is_uptrend and close[i] > highest_high[i] and volume_filter[i]:
            signals[i] = 0.25
        # Short entry: breakdown below Donchian low in downtrend with volume
        elif not is_uptrend and close[i] < lowest_low[i] and volume_filter[i]:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0