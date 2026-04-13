#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily volume confirmation and daily EMA trend filter.
# The strategy buys when price breaks above the 20-period Donchian high with
# above-average volume and price above daily EMA (trend alignment).
# It sells when price breaks below the 20-period Donchian low with
# above-average volume and price below daily EMA.
# Exit when price crosses the Donchian midline (10-period average).
# Uses daily EMA to avoid counter-trend trades and volume to confirm breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volume and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily average volume (20-period) for confirmation
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # Daily EMA trend filter (21-period)
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (21 + 1)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * ema_multiplier + ema_1d[i-1]
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    donch_mid = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
        donch_mid[i] = (donch_high[i] + donch_low[i]) / 2
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(donch_mid[i]) or np.isnan(avg_vol_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_vol_1d_aligned[i]
        daily_ema = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x daily average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + above daily EMA
            if (price > donch_high[i] and volume_confirm and price > daily_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + below daily EMA
            elif (price < donch_low[i] and volume_confirm and price < daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if price < donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if price > donch_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0