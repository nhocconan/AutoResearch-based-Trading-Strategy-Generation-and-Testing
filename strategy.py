#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Uses Donchian breakout on 4h with 1d EMA200 trend filter and volume confirmation. 
# Enters long when price breaks above Donchian(20) high AND close > 1d EMA200 AND volume > 1.5x average volume.
# Enters short when price breaks below Donchian(20) low AND close < 1d EMA200 AND volume > 1.5x average volume.
# Exits when price crosses back across Donchian midline or trend filter fails.
# Designed for 20-40 trades/year on 4h to avoid fee drag. Works in bull/bear via trend-following with strong filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # 1-day data for EMA200 filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.zeros_like(high_4h)
    donchian_low = np.zeros_like(low_4h)
    donchian_mid = np.zeros_like(high_4h)
    
    for i in range(len(high_4h)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
            donchian_mid[i] = np.nan
        else:
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # 1-day EMA200
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Average volume for confirmation (20-period)
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Align 4h indicators to 4h timeframe (no alignment needed as we're already on 4h)
    # But we need to align 1d EMA200 to 4h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Ensure EMA200 is ready
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend alignment filter
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midline OR trend filter fails
            if close[i] < donchian_mid[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midline OR trend filter fails
            if close[i] > donchian_mid[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high AND trend filter AND volume confirmation
            if close[i] > donchian_high[i] and price_above_ema and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low AND trend filter AND volume confirmation
            elif close[i] < donchian_low[i] and price_below_ema and volume_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals