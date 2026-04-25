#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Weekly EMA Trend and Volume Spike
Hypothesis: Daily Donchian breakouts capture strong trending moves. Weekly EMA filter ensures
alignment with higher timeframe trend, reducing whipsaws in ranging markets. Volume confirmation
avoids false breakouts. Designed to work in both bull and bear markets by allowing long and short
entries. Targets 30-100 trades over 4 years (7-25/year) to minimize fee drag.
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
    
    # Calculate 34-period EMA on 1w close
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for 1w volume spike
    vol_ma_20_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        vol_ma_20_1w[i] = np.mean(df_1w['volume'].values[i-19:i+1])
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 1d volume spike
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma_20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ma_1w = vol_ma_20_1w_aligned[i]
        vol_ma_1d = vol_ma_20_1d[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Volume confirmation: current 1d volume > 2.0 * 20-period 1d average
        volume_confirm = curr_volume > 2.0 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian channel, above weekly EMA, volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian channel, below weekly EMA, volume confirmation
            short_entry = (curr_close < lower_channel and 
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
            if curr_close < lower_channel or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian channel OR above weekly EMA
            if curr_close > upper_channel or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0