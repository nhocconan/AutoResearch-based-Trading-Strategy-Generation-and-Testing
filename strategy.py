#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND close > 1d EMA34 AND volume > 1.5x 20-period average volume.
# Short when: price breaks below Donchian(20) low AND close < 1d EMA34 AND volume > 1.5x 20-period average volume.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 12-37 trades/year.
# Donchian channels provide clear structure; 1d EMA34 filters for higher timeframe trend alignment;
# volume confirmation ensures breakouts have conviction. Works in bull (breakouts with trend) and bear
# (breakouts against trend filtered by 1d EMA) by requiring volume spike and trend alignment.

name = "12h_Donchian20_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 20-period average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian and volume average
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_avg_volume = avg_volume[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = curr_volume > 1.5 * curr_avg_volume
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high AND close > 1d EMA34 AND volume confirmed
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_34 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND close < 1d EMA34 AND volume confirmed
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_34 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: close below Donchian low OR close < 1d EMA34
            if (curr_close < curr_lowest_low or 
                curr_close < curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: close above Donchian high OR close > 1d EMA34
            if (curr_close > curr_highest_high or 
                curr_close > curr_ema_34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals