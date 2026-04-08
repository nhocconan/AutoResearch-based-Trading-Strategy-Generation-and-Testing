#!/usr/bin/env python3
# 12h_1d_1w_donchian_breakout_volume_regime_v2
# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w trend filter.
# Long when price breaks above Donchian high + volume spike + 1w uptrend.
# Short when price breaks below Donchian low + volume spike + 1w downtrend.
# Uses volume confirmation to avoid false breakouts and weekly trend to align with higher timeframe momentum.
# Designed for 12-30 trades/year on 12h to avoid fee drag. Works in bull/bear via multi-timeframe alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_donchian_breakout_volume_regime_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i+1])
        donchian_low[i] = np.min(low[i-20:i+1])
    
    # Volume spike: current volume > 1.5 * 20-period average volume
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i+1])
    volume_spike = volume > 1.5 * vol_ma
    
    # Get 1d data for additional volume confirmation (optional, can use same logic)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(vol_1d[i-20:i+1])
    volume_spike_1d = vol_1d > 1.5 * vol_ma_1d
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 1w EMA25 for trend filter
    ema25_1w = pd.Series(close_1w).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_1w_aligned = align_htf_to_ltf(prices, df_1w, ema25_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Donchian needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema25_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Use either 12h or 1d volume spike (12h primary, 1d as confirmation)
        vol_spike = volume_spike[i] or volume_spike_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or volume spike fails (optional)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + volume spike + 1w uptrend
            if (close[i] > donchian_high[i] and 
                vol_spike and 
                close[i] > ema25_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume spike + 1w downtrend
            elif (close[i] < donchian_low[i] and 
                  vol_spike and 
                  close[i] < ema25_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals