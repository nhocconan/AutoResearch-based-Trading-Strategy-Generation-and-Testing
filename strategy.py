#!/usr/bin/env python3
# 4h_donchian_breakout_1w_trend_filter_v1
# Hypothesis: Uses Donchian(20) breakout on 4h with 1-week EMA50 trend filter to capture sustained trends in both bull and bear markets. Volume confirmation reduces false breakouts. Designed for ~25-40 trades/year on 4h to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1w_trend_filter_v1"
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
    
    # 1-week data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Calculate 1-week EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4-period average volume for confirmation
    avg_vol_4 = np.full_like(volume, np.nan)
    for i in range(4, len(volume)):
        avg_vol_4[i] = np.mean(volume[i-4:i])
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    avg_vol_4_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_vol_4_aligned[i]) or np.isnan(volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * avg_vol_4_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or below 1w EMA50
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or above 1w EMA50
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price closes above Donchian high with volume confirmation and above 1w EMA50
            if close[i] > donchian_high_aligned[i] and volume_confirm and close[i] > ema50_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below Donchian low with volume confirmation and below 1w EMA50
            elif close[i] < donchian_low_aligned[i] and volume_confirm and close[i] < ema50_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals