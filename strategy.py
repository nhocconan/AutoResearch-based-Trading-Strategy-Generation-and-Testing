#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with 1w EMA34 Trend and Volume Spike
Hypothesis: Daily Donchian(20) breakouts capture medium-term trends. 
1w EMA34 filter ensures we trade with the weekly trend. Volume confirmation 
avoids false breakouts. Works in both bull and bear markets by being 
directional (long in uptrend, short in downtrend). Targets 30-100 trades 
over 4 years (7-25/year) on 1d timeframe.
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
    
    # Get 1d data for Donchian channels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(
        window=20, min_periods=20
    ).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(
        window=20, min_periods=20
    ).min().values
    
    # Align Donchian levels to 1d (previous day's levels available after 1d close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Get 1w data for EMA34 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1w close for trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-period volume MA for 1d volume confirmation
    vol_ma_20_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_1d[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_1d[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_ma_1d = vol_ma_20_1d[i]
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_1d
        
        if position == 0:
            # Look for entry signals
            # Long: price > EMA34 (uptrend) AND breaks above Donchian high AND volume confirmation
            long_entry = (curr_close > ema_trend and 
                         curr_high > donchian_high and 
                         volume_confirm)
            # Short: price < EMA34 (downtrend) AND breaks below Donchian low AND volume confirmation
            short_entry = (curr_close < ema_trend and 
                          curr_low < donchian_low and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below EMA34 OR breaks below Donchian low (failed breakout)
            if curr_close < ema_trend or curr_low < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above EMA34 OR breaks above Donchian high (failed breakdown)
            if curr_close > ema_trend or curr_high > donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0