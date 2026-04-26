#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Donchian(20) breakout on 12h with 1w EMA50 trend filter and volume spike (>2x average volume).
In bull markets: price breaks above upper Donchian with 1w uptrend and high volume → long.
In bear markets: price breaks below lower Donchian with 1w downtrend and high volume → short.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
Requires BTC/ETH edge via 1w trend and volume filters; avoids SOL-only bias by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian calculation, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Hold current position by default
        if position == 0:
            signals[i] = 0.0
        elif position == 1:
            signals[i] = base_size
        else:
            signals[i] = -base_size
        
        # Skip if any data not ready
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(avg_volume[i]):
            continue
            
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above upper Donchian with 1w uptrend and volume confirmation
        long_condition = (close_val > upper_channel) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below lower Donchian with 1w downtrend and volume confirmation
        short_condition = (close_val < lower_channel) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite Donchian level break
        exit_long = (close_val < ema_val) or (close_val < lower_channel)
        exit_short = (close_val > ema_val) or (close_val > upper_channel)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        # else: hold position (already set above)
    
    return signals

name = "12h_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0