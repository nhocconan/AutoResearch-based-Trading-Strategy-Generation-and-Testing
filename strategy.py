#!/usr/bin/env python3
"""
12h Volume Spike + 1d EMA34 Trend + 4h Donchian(20) Breakout
Hypothesis: Volume spikes confirm institutional interest. Combined with 1d EMA34 trend filter and 4h Donchian breakout, this captures strong momentum moves while minimizing false signals. The 12h timeframe reduces trade frequency to avoid fee drag. Works in bull/bear markets via trend filter - only trades in direction of 1d trend.
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
    
    # Get 4h data for Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian(20) channels
    donch_high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for EMA34 warmup and Donchian calculation
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        upper_donch = donch_high_aligned[i]
        lower_donch = donch_low_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper channel AND above 1d EMA34 (uptrend filter) with volume spike
            long_condition = (curr_close > upper_donch) and (curr_close > ema_trend) and volume_spike
            # Short: price breaks below 4h Donchian lower channel AND below 1d EMA34 (downtrend filter) with volume spike
            short_condition = (curr_close < lower_donch) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: price returns below 4h Donchian lower channel or trend breaks
            if curr_close < lower_donch or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above 4h Donchian upper channel or trend breaks
            if curr_close > upper_donch or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolumeSpike_1dEMA34_Trend_4hDonchian20_Breakout_v1"
timeframe = "12h"
leverage = 1.0