#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 80-120 total trades over 4 years (20-30/year).
# Donchian breakout captures trending moves. 1d EMA50 filters for major trend alignment.
# Volume spike ensures institutional participation. Works in bull via breakout longs, in bear via breakdown shorts.

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 12h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 30, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: price breaks above Donchian upper channel AND above 1d EMA50
                if (curr_close > curr_highest_20 and 
                    curr_close > curr_ema_50_1d):
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian lower channel AND below 1d EMA50
                elif (curr_close < curr_lowest_20 and 
                      curr_close < curr_ema_50_1d):
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: price closes below Donchian lower channel (breakdown)
            if curr_close < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper channel (breakout)
            if curr_close > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals