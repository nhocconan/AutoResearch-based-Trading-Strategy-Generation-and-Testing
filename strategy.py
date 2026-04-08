#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: Donchian breakout on 4h with 1d trend filter and volume confirmation. Works in bull/bear via trend alignment and volume filter.
# Uses 1d EMA50 for trend direction and 4h volume spike for entry confirmation. Designed for ~25 trades/year on 4h to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4-hour data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    donchian_high = np.zeros_like(high_4h)
    donchian_low = np.zeros_like(low_4h)
    
    for i in range(len(high_4h)):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high_4h[i-20:i])
            donchian_low[i] = np.min(low_4h[i-20:i])
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detector (20-period average)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_spike = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend fails
            if close[i] < donchian_low_aligned[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend fails
            if close[i] > donchian_high_aligned[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high + volume spike + uptrend
            if close[i] > donchian_high_aligned[i] and volume_spike and price_above_ema:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low + volume spike + downtrend
            elif close[i] < donchian_low_aligned[i] and volume_spike and price_below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals