#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation
# Long when: price breaks above 12h Donchian upper(20) AND price > 1d EMA34 (uptrend) AND volume > 2.0x 20-period avg volume
# Short when: price breaks below 12h Donchian lower(20) AND price < 1d EMA34 (downtrend) AND volume > 2.0x 20-period avg volume
# Uses Donchian channels for structure, 1d EMA for trend filter (avoid counter-trend), volume spike for breakout validity.
# Designed for 12h timeframe to target 12-37 trades/year with discrete sizing (0.25) to minimize fee drag.
# Works in bull/bear via trend filter + volatility expansion requirement.

name = "12h_Donchian20_EMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    donchian_upper = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_12h.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Volume spike: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available (first bars)
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_donchian_upper = donchian_upper_aligned[i]
        curr_donchian_lower = donchian_lower_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price falls below Donchian lower (mean reversion)
            # 2. Price falls below 1d EMA34 (trend change)
            if (curr_close < curr_donchian_lower or
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price rises above Donchian upper (mean reversion)
            # 2. Price rises above 1d EMA34 (trend change)
            if (curr_close > curr_donchian_upper or
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND above 1d EMA34 AND volume spike
            if (curr_close > curr_donchian_upper and
                curr_close > curr_ema_34_1d and
                curr_volume_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower AND below 1d EMA34 AND volume spike
            elif (curr_close < curr_donchian_lower and
                  curr_close < curr_ema_34_1d and
                  curr_volume_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals