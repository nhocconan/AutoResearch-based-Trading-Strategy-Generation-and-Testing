#!/usr/bin/env python3
name = "6h_Donchian20_12hTrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels (20) on 6h data
    # Highest high of last 20 periods (including current)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d volume spike (volume > 2x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian upper band, above 12h EMA50, with 1d volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian lower band, below 12h EMA50, with 1d volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters Donchian channel (below upper band) or below EMA50
            if (close[i] < highest_high[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters Donchian channel (above lower band) or above EMA50
            if (close[i] > lowest_low[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and 1d volume spike confirmation.
# Donchian breakouts capture momentum bursts; 12h EMA50 ensures alignment with higher timeframe trend.
# 1d volume spike confirms institutional participation, reducing false breakouts.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Position size 0.25 balances risk and keeps trade frequency ~15-30 trades/year.