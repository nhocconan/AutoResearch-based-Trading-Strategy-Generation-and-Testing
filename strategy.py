#!/usr/bin/env python3
name = "1d_WeeklyDonchian20_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly Donchian (20) channels ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian upper/lower (20 periods)
    high_max_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donchian_upper_20 = high_max_20
    donchian_lower_20 = low_min_20
    
    # Weekly trend filter: close above/below 50 EMA
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly volume spike filter: volume > 2x 20-period average
    vol_1w = df_1w['volume'].values
    vol_avg_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = vol_1w > (2.0 * vol_avg_1w)
    
    # Align to daily timeframe
    donchian_upper_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_20)
    donchian_lower_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_20_aligned[i]) or 
            np.isnan(donchian_lower_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above weekly Donchian upper + above weekly EMA50 + volume spike
            if (close[i] > donchian_upper_20_aligned[i] and
                close[i] > ema50_1w_aligned[i] and
                vol_spike_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below weekly Donchian lower + below weekly EMA50 + volume spike
            elif (close[i] < donchian_lower_20_aligned[i] and
                  close[i] < ema50_1w_aligned[i] and
                  vol_spike_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below weekly Donchian lower or below weekly EMA50
            if close[i] < donchian_lower_20_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above weekly Donchian upper or above weekly EMA50
            if close[i] > donchian_upper_20_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals