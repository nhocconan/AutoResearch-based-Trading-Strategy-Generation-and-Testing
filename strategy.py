# [40286] Hypothesis: 4h Donchian breakout with volume confirmation and 1-day VWAP trend filter.
# Uses breakout of 20-period Donchian channel (from 4h) confirmed by volume > 20-period average volume.
# Trend filter: price above/below 1-day VWAP (volume-weighted average price) to avoid counter-trend trades.
# Designed for low trade frequency (~20-50/year) to minimize fee drag and work in both bull/bear markets.

#!/usr/bin/env python3
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
    
    # Get 4h data for Donchian and volume calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate 20-period Donchian channels on 4h
    donchian_high = np.full(len(high_4h), np.nan)
    donchian_low = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Calculate 20-period average volume on 4h
    avg_volume_4h = np.full(len(vol_4h), np.nan)
    for i in range(20, len(vol_4h)):
        avg_volume_4h[i] = np.mean(vol_4h[i-20:i])
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP components
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    tpv_1d = typical_price_1d * vol_1d
    
    # Calculate 20-period cumulative sums for VWAP
    cum_tpv_1d = np.full(len(tpv_1d), np.nan)
    cum_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(len(tpv_1d)):
        if i == 0:
            cum_tpv_1d[i] = tpv_1d[i]
            cum_vol_1d[i] = vol_1d[i]
        else:
            cum_tpv_1d[i] = cum_tpv_1d[i-1] + tpv_1d[i]
            cum_vol_1d[i] = cum_vol_1d[i-1] + vol_1d[i]
    
    # Calculate VWAP with 20-period window
    vwap_1d = np.full(len(tpv_1d), np.nan)
    for i in range(19, len(tpv_1d)):
        start_idx = i - 19
        sum_tpv = cum_tpv_1d[i] - (cum_tpv_1d[start_idx-1] if start_idx > 0 else 0)
        sum_vol = cum_vol_1d[i] - (cum_vol_1d[start_idx-1] if start_idx > 0 else 0)
        if sum_vol > 0:
            vwap_1d[i] = sum_tpv / sum_vol
    
    # Align all indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    avg_volume_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_volume_4h)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_4h_aligned[i]) or
            np.isnan(vwap_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > average volume
        vol_confirm = volume[i] > avg_volume_4h_aligned[i]
        
        # Donchian breakout conditions
        donchian_breakout_long = close[i] > donchian_high_aligned[i]
        donchian_breakout_short = close[i] < donchian_low_aligned[i]
        
        # VWAP trend filter: price above VWAP for long, below for short
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Entry conditions with confluence
        long_entry = donchian_breakout_long and vol_confirm and price_above_vwap
        short_entry = donchian_breakout_short and vol_confirm and price_below_vwap
        
        # Exit conditions: opposite Donchian breakout or VWAP cross
        exit_long = position == 1 and (donchian_breakout_short or close[i] < vwap_1d_aligned[i])
        exit_short = position == -1 and (donchian_breakout_long or close[i] > vwap_1d_aligned[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_vwap_volume"
timeframe = "4h"
leverage = 1.0