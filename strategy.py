#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with weekly EMA50 trend filter and volume confirmation (>2x average volume). 
In bull markets: price breaks above upper Donchian with weekly uptrend and high volume → long. 
In bear markets: price breaks below lower Donchian with weekly downtrend and high volume → short. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
Requires BTC/ETH edge via weekly trend and volume filters; avoids SOL-only bias by requiring trend alignment.
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
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Calculate Donchian levels using previous 20 days' high/low
        # We need lookback of 20 days excluding current bar
        lookback_start = i - 20
        if lookback_start < 0:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
            
        # Donchian channels: highest high and lowest low of past 20 days
        highest_high = np.max(high[lookback_start:i])
        lowest_low = np.min(low[lookback_start:i])
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(highest_high) or np.isnan(lowest_low) or np.isnan(ema_val) or np.isnan(avg_vol):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 2.0x average volume (stricter to reduce trades)
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above upper Donchian with weekly uptrend and volume confirmation
        long_condition = (close_val > highest_high) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below lower Donchian with weekly downtrend and volume confirmation
        short_condition = (close_val < lowest_low) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal or opposite breakout
        exit_long = (close_val < ema_val) or (close_val < lowest_low)
        exit_short = (close_val > ema_val) or (close_val > highest_high)
        
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
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0