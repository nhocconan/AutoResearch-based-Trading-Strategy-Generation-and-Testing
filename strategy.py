#!/usr/bin/env python3
"""
12h Donchian Channel Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Donchian(20) breakouts capture momentum in trending markets. Combined with 1d EMA34 trend filter and volume confirmation, this strategy avoids false breakouts in ranging markets. The 12h timeframe reduces trade frequency to minimize fee drag while capturing multi-day swings. Works in bull markets via upward breakouts and bear markets via downward breakouts with trend filter.
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
    entry_price = 0.0
    
    # Start index: need enough for Donchian(20) and EMA34 warmup
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if EMA data not ready
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
        
        # Donchian Channel (20): highest high and lowest low of past 20 bars (excluding current)
        if i >= 20:
            highest_high = np.max(high[i-20:i])  # past 20 bars, not including current
            lowest_low = np.min(low[i-20:i])
        else:
            highest_high = np.max(high[:i]) if i > 0 else curr_high
            lowest_low = np.min(low[:i]) if i > 0 else curr_low
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA34 (uptrend filter)
            long_condition = (curr_close > highest_high) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below Donchian low AND below 1d EMA34 (downtrend filter)
            short_condition = (curr_close < lowest_low) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below Donchian low or trend breaks
            if curr_close < lowest_low or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian high or trend breaks
            if curr_close > highest_high or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0