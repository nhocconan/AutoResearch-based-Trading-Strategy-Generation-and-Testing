#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Weekly EMA34 Trend and Volume Spike
Hypothesis: Daily Donchian channel breakouts capture strong trends. Weekly EMA34 filter ensures alignment with higher timeframe trend, reducing false breakouts in choppy markets. Volume confirmation adds conviction. Designed for low trade frequency (target: 30-100 trades over 4 years) to minimize fee drag in BTC/ETH markets. Works in both bull and bear markets by following the weekly trend direction.
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
    
    # Get weekly data for EMA trend (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 34-period EMA on weekly close
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period Donchian channels on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: upper = max(high, lookback=20), lower = min(low, lookback=20)
    donchian_upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned via get_htf_data)
    # No additional alignment needed as we're working on 1d timeframe
    
    # Calculate 20-period volume MA for 1d volume spike
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian and volume MA
    start_idx = max(20, 34)  # 20 for Donchian, 34 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_upper_1d[i]) or np.isnan(donchian_lower_1d[i]) or
            np.isnan(vol_ma_20_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1w_aligned[i]
        upper_channel = donchian_upper_1d[i]
        lower_channel = donchian_lower_1d[i]
        vol_ma_1d = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper Donchian, above weekly EMA, volume confirmation
            long_entry = (curr_close > upper_channel and 
                         curr_close > ema_trend and 
                         volume_confirm)
            # Short: price breaks below lower Donchian, below weekly EMA, volume confirmation
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
            # Exit: price falls below lower Donchian OR below weekly EMA
            if curr_close < lower_channel or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above upper Donchian OR above weekly EMA
            if curr_close > upper_channel or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_WeeklyEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0