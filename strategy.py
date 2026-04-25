#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with 1d EMA34 Trend and Volume Confirmation
Hypothesis: Donchian channel breakouts capture strong momentum. 
Combined with 1d EMA34 trend filter and volume confirmation to avoid false breakouts.
Works in bull markets (trend continuation) and bear markets (breakdowns with volume).
Target: 20-50 trades/year on 4h timeframe to minimize fee drag.
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
    
    # Get 1d data for EMA trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period Donchian channels on 4h
    # Upper = max(high over last 20 periods), Lower = min(low over last 20 periods)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    for i in range(20, n):
        donchian_upper[i] = np.max(high[i-19:i+1])
        donchian_lower[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for 4h volume confirmation
    vol_ma_20_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = donchian_upper[i]
        lower_channel = donchian_lower[i]
        vol_ma_4h = vol_ma_20_4h[i]
        
        # Volume confirmation: current 4h volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma_4h
        
        if position == 0:
            # Look for entry signals
            # Long: price > EMA34 (uptrend) AND breaks above upper Donchian AND volume confirmation
            long_entry = (curr_close > ema_trend and 
                         curr_high > upper_channel and 
                         volume_confirm)
            # Short: price < EMA34 (downtrend) AND breaks below lower Donchian AND volume confirmation
            short_entry = (curr_close < ema_trend and 
                          curr_low < lower_channel and 
                          volume_confirm)
            
            if long_entry:
                signals[i] = 0.30
                position = 1
            elif short_entry:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below EMA34 OR breaks below lower Donchian (failed breakout)
            if curr_close < ema_trend or curr_low < lower_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: price rises above EMA34 OR breaks above upper Donchian (failed breakdown)
            if curr_close > ema_trend or curr_high > upper_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0