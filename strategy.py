#!/usr/bin/env python3
"""
6h Donchian(20) breakout + 12h EMA50 trend + volume confirmation
Hypothesis: Donchian channel breakouts capture sustained moves. 12h EMA50 filters for higher-timeframe trend alignment.
Volume confirmation ensures institutional participation. Works in bull/bear via EMA50 trend filter.
Target: 12-37 trades/year on 6h timeframe.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and EMA50 to propagate
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if EMA50 data not ready
        if np.isnan(ema_50_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50 = ema_50_aligned[i]
        
        # Donchian(20): 20-period high/low
        if i >= 20:
            donchian_high = np.max(high[i-19:i+1])
            donchian_low = np.min(low[i-19:i+1])
        else:
            donchian_high = np.max(high[:i+1])
            donchian_low = np.min(low[:i+1])
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_confirmed = curr_volume > 1.5 * vol_ma_20
        
        # Trend filter: price above/below 12h EMA50
        uptrend = curr_close > ema_50
        downtrend = curr_close < ema_50
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmed AND uptrend
            long_condition = (curr_high > donchian_high) and volume_confirmed and uptrend
            # Short: price breaks below Donchian low AND volume confirmed AND downtrend
            short_condition = (curr_low < donchian_low) and volume_confirmed and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below Donchian low or trend changes
            if curr_close <= donchian_low or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend changes
            if curr_close >= donchian_high or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0