#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 4h timeframe for signal generation with Donchian channel breakouts
# 1d EMA50 provides trend filter to avoid counter-trend trades
# Volume confirmation (2.0x 20-period average) ensures institutional participation
# Designed for low trade frequency (~25-50 trades/year) to minimize fee drag
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Named: 4h_Donchian20_1dEMA50_VolumeSpike_v1

name = "4h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Donchian channels (20-period)
        # Highest high and lowest low over last 20 periods (including current)
        highest_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lowest_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        if np.isnan(highest_high) or np.isnan(lowest_low):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (2.0x 20-period average)
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (vol_ma * 2.0)
        else:
            volume_confirm = False
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel + price > 1d EMA50 + volume confirm
            if close[i] > highest_high and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel + price < 1d EMA50 + volume confirm
            elif close[i] < lowest_low and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower channel (stop and reverse) or time-based exit
            if close[i] < lowest_low:
                signals[i] = -0.25  # Reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper channel (stop and reverse) or time-based exit
            if close[i] > highest_high:
                signals[i] = 0.25  # Reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals