#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily EMA Trend and Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum moves. Combined with daily EMA trend filter
and volume confirmation, this strategy works in both bull and bear markets by only taking breakouts
in the direction of the higher timeframe trend. 12h timeframe targets 50-150 trades over 4 years to
minimize fee drag while maintaining sufficient sample size.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close (only needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel, above 1d EMA, volume confirmation
            long_entry = (curr_high > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian channel, below 1d EMA, volume confirmation
            short_entry = (curr_low < lower_channel and 
                          curr_close < ema_trend and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below lower Donchian channel OR below 1d EMA
            if curr_low < lower_channel or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR above 1d EMA
            if curr_high > upper_channel or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_DailyEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0