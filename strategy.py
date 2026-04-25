#!/usr/bin/env python3
"""
12h Donchian Channel Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Donchian(20) breakouts capture strong trends. Weekly EMA(50) filter ensures alignment with major trend.
Volume spike (>1.5x 20-period MA) confirms momentum. Designed for 12h timeframe to target 50-150 trades over 4 years.
Works in bull markets (breakouts above upper band) and bear markets (breakouts below lower band with short bias).
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
    
    # Get 1w data for EMA trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1w close (only needs completed 1w candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on 12h
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-19:i+1])
        lower_channel[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 12h volume spike
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(vol_ma_20_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_1w_aligned[i]
        upper_chan = upper_channel[i]
        lower_chan = lower_channel[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_12h
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel, above weekly EMA, volume confirmation
            long_entry = (curr_close > upper_chan and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian channel, below weekly EMA, volume confirmation
            short_entry = (curr_close < lower_chan and 
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
            # Exit: price falls below lower Donchian channel OR below weekly EMA
            if curr_close < lower_chan or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR above weekly EMA
            if curr_close > upper_chan or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_WeeklyEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0