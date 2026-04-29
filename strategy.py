#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses Donchian channel breakouts for structural entry/exit
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume > 1.5x 20-period average confirms breakout strength
# Discrete position sizing (0.25) to manage risk and reduce fee churn
# Designed for lower trade frequency (target: 12-37/year) to overcome fee drag in bear markets

name = "12h_Donchian20_1dEMA50_VolumeConfirm_v1"
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
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period Donchian channels (highest high, lowest low)
    # Using 20-period lookback for breakout signals
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Donchian and volume warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low (Donchian mean reversion)
            if curr_low < curr_lowest_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high (Donchian mean reversion)
            if curr_high > curr_highest_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when price breaks above 20-period high, 1d EMA50 up-trend, volume confirmed
            if curr_high > curr_highest_high and curr_close > curr_ema50_1d and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-period low, 1d EMA50 down-trend, volume confirmed
            elif curr_low < curr_lowest_low and curr_close < curr_ema50_1d and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals