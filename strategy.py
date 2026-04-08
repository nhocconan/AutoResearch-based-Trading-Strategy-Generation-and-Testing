#!/usr/bin/env python3
# 4h_donchian_breakout_1d_volume_filter_v1
# Hypothesis: Uses Donchian channel breakout (20-period) on 4h for entry, confirmed by 1d volume spike (>1.5x 20-period average) and price above/below 1d EMA50 for trend filter. Exits on opposite Donchian breakout or volume drop. Designed for 20-40 trades/year on 4h to avoid fee drag. Works in bull/bear via trend-following with volume and trend filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_filter_v1"
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
    
    # 4-hour data for Donchian channel
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 1-day data for EMA50 and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channel (20-period) on 4h
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # 1-day EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1-day volume average (20-period)
    vol_ma_1d = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align 4h Donchian bands and 1d indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-period average
        volume_spike = volume_1d[i] > 1.5 * vol_ma_1d[i] if not np.isnan(vol_ma_1d[i]) else False
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or volume spike ends
            if close[i] < donchian_low_aligned[i] or not volume_spike:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or volume spike ends
            if close[i] > donchian_high_aligned[i] or not volume_spike:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above Donchian high, volume spike, and price above EMA50
            if close[i] > donchian_high_aligned[i] and volume_spike and price_above_ema:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below Donchian low, volume spike, and price below EMA50
            elif close[i] < donchian_low_aligned[i] and volume_spike and price_below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals