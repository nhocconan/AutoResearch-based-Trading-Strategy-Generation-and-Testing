#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeSpike
Hypothesis: 6h Donchian(20) breakouts aligned with 12h EMA50 trend and 1d volume spikes capture sustained moves while avoiding chop. The 12h EMA50 provides a medium-term trend filter, and 1d volume spikes (>2.0x 24-period average) confirm institutional participation. Discrete sizing (0.25) limits fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume spike: current volume > 2.0 * 24-period average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=24, min_periods=24).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    
    # Align all indicators to primary timeframe (6h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 12h EMA50 (50), Donchian (20), 1d volume avg (24)
    start_idx = max(50, 20, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_12h_val = ema_50_12h_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Determine trend: price relative to 12h EMA50
            is_uptrend = close_val > ema_12h_val
            is_downtrend = close_val < ema_12h_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume spikes
                if (close_val > upper) and vol_spike:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume spikes
                if (close_val < lower) and vol_spike:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches Donchian low or trend changes to downtrend
            exit_condition = (close_val < lower) or (close_val < ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches Donchian high or trend changes to uptrend
            exit_condition = (close_val > upper) or (close_val > ema_12h_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeSpike"
timeframe = "6h"
leverage = 1.0