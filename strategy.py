#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation
Hypothesis: Donchian breakouts capture momentum bursts while the 1d EMA34 filter ensures alignment with higher timeframe trend, working in both bull and bear markets. Volume confirmation filters false breakouts. 4h timeframe targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian(20) and EMA34 warmup
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        
        # Donchian(20) channels: highest high and lowest low of last 20 periods
        highest_20 = np.max(high[i-19:i+1])
        lowest_20 = np.min(low[i-19:i+1])
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_confirm = curr_volume > 1.5 * vol_ma_20
        
        # Breakout signals with trend filter
        if position == 0:
            # Long: price breaks above Donchian upper band AND above daily EMA34 (uptrend filter)
            long_condition = (curr_close > highest_20) and (curr_close > ema_trend) and volume_confirm
            # Short: price breaks below Donchian lower band AND below daily EMA34 (downtrend filter)
            short_condition = (curr_close < lowest_20) and (curr_close < ema_trend) and volume_confirm
            
            if long_condition:
                signals[i] = 0.30
                position = 1
            elif short_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian lower band or trend breaks
            if curr_close < lowest_20 or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price returns to Donchian upper band or trend breaks
            if curr_close > highest_20 or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0