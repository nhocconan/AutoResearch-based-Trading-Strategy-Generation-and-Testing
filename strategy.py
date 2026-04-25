#!/usr/bin/env python3
"""
1h Donchian(20) Breakout + 4h EMA50 Trend + 1d Volume Spike
Hypothesis: On 1h timeframe, Donchian breakouts capture short-term momentum. 
Filter by 4h EMA50 for medium-term trend alignment and 1d volume spike for 
institutional participation. Works in bull markets (breakouts with trend) and 
bear markets (failed breaks retesting Donchian bands). Session filter (08-20 UTC) 
reduces noise. Target 15-30 trades/year to avoid fee drag on 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # 4h data for EMA50 trend (loaded ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for volume spike (loaded ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20) + 5  # EMA50 + Donchian20 + buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume spike condition: current volume > 2.0 * 20-period 1d average
        volume_spike = curr_volume > (vol_ma_20_1d_aligned[i] * 2.0)
        
        if position == 0:
            # Look for entry signals - require: Donchian breakout + volume spike + 4h EMA50 trend alignment
            long_breakout = curr_close > donchian_high[i]
            short_breakout = curr_close < donchian_low[i]
            
            long_entry = long_breakout and volume_spike and (curr_close > ema_50_4h_aligned[i])
            short_entry = short_breakout and volume_spike and (curr_close < ema_50_4h_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Donchian low retrace or trend change
            if curr_close < donchian_low[i] or curr_close < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on Donchian high retrace or trend change
            if curr_close > donchian_high[i] or curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_Breakout_4hEMA50_Trend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0