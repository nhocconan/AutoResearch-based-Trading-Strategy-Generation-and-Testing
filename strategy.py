#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when price breaks above 6h Donchian upper channel AND 1w trend is up (close > EMA20) AND volume spike.
# Short when price breaks below 6h Donchian lower channel AND 1w trend is down (close < EMA20) AND volume spike.
# Uses weekly EMA for trend filter and 6h volume spike for momentum confirmation.
# Designed for 15-30 trades/year to minimize fee drag while capturing strong trends.
name = "6h_Donchian20_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = df_1w['close'].values > ema_20_1w
    trend_down_1w = df_1w['close'].values < ema_20_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Load 1d data for volume spike (more reliable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: break above Donchian upper, weekly uptrend, volume spike
            long_condition = (close[i] > highest_high_20[i]) and trend_up_1w_aligned[i] and vol_spike_1d_aligned[i]
            # Short condition: break below Donchian lower, weekly downtrend, volume spike
            short_condition = (close[i] < lowest_low_20[i]) and trend_down_1w_aligned[i] and vol_spike_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian lower or weekly trend turns down
            if (close[i] < lowest_low_20[i]) or (not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian upper or weekly trend turns up
            if (close[i] > highest_high_20[i]) or (not trend_down_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals