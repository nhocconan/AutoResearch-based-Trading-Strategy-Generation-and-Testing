#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian(20) breakout with 4h trend filter (EMA50) and 1d volume spike filter.
Long when price breaks above Donchian(20) high AND 4h EMA50 rising AND 1d volume > 1.5 * 20-day average volume.
Short when price breaks below Donchian(20) low AND 4h EMA50 falling AND 1d volume > 1.5 * 20-day average volume.
Exit when price touches opposite Donchian(20) level or volume condition fails.
Uses 4h for trend direction (EMA50), 1d for volume confirmation, 1h for entry timing and Donchian channels.
Target: 80-120 total trades over 4 years (20-30/year) to stay within fee drag limits.
Session filter: 08-20 UTC to avoid low-liquidity periods.
Position size: 0.20 (20% of capital) to limit drawdown.
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
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d 20-period average volume and volume spike condition
    avg_vol_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_vol_20d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike, additional_delay_bars=0)
    
    # Calculate 1h Donchian channels (20-period)
    # Using rolling window for highest high and lowest low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for Donchian(20) and EMA50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]   # break below previous period's low
        
        # 4h EMA50 trend direction (rising/falling)
        ema50_now = ema50_4h_aligned[i]
        ema50_prev = ema50_4h_aligned[i-1]
        ema50_rising = ema50_now > ema50_prev
        ema50_falling = ema50_now < ema50_prev
        
        # 1d volume spike condition
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Long: Donchian breakout up AND 4h EMA50 rising AND 1d volume spike
            if breakout_up and ema50_rising and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: Donchian breakout down AND 4h EMA50 falling AND 1d volume spike
            elif breakout_down and ema50_falling and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Donchian breakout down OR volume spike fails
            if breakout_down or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Donchian breakout up OR volume spike fails
            if breakout_up or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0