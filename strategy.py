#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_1dVolume_1wTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    lookback = 20
    
    donchian_high = np.full(len(high_1d), np.nan)
    donchian_low = np.full(len(low_1d), np.nan)
    
    for i in range(lookback - 1, len(high_1d)):
        donchian_high[i] = np.max(high_1d[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low_1d[i - lookback + 1:i + 1])
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):
        vol_ma[i] = np.mean(vol_1d[i - 19:i + 1])
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 / 35) + (ema_34_1w[i-1] * 33 / 35)
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(vol_ma_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume spike condition: current 1d volume > 1.5 * 20-day average
        # Get current day's volume from 1d data
        current_day_idx = i // 16  # 16 four-hour bars per day
        if current_day_idx < len(vol_1d) and current_day_idx < len(vol_ma):
            vol_current = vol_1d[current_day_idx]
            vol_average = vol_ma[current_day_idx]
            volume_spike = vol_current > (vol_average * 1.5)
        else:
            volume_spike = False
        
        # Trend filter: price above/below 1w EMA34
        price_above_1w_ema = close[i] > ema_34_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long when price breaks above Donchian high + volume spike + uptrend
            if close[i] > donchian_high_aligned[i] and volume_spike and price_above_1w_ema:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low + volume spike + downtrend
            elif close[i] < donchian_low_aligned[i] and volume_spike and price_below_1w_ema:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals