#!/usr/bin/env python3
"""
12-hour Donchian Breakout with 1-Day Trend Filter and Volume Confirmation
Long when price breaks above Donchian(20) high on 1-day timeframe (trend up),
short when price breaks below Donchian(20) low (trend down), with volume filter.
Trades only on 12h closes to limit frequency. Target: 20-40 trades/year.
Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily Donchian channels (20-period)
    donch_high = np.full_like(close_1d, np.nan)
    donch_low = np.full_like(close_1d, np.nan)
    for i in range(19, len(high_1d)):
        donch_high[i] = np.max(high_1d[i-19:i+1])
        donch_low[i] = np.min(low_1d[i-19:i+1])
    
    # Daily trend: EMA(50) slope
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = np.zeros_like(ema_50)
    ema_slope[1:] = ema_50[1:] - ema_50[:-1]
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_slope_12h = align_htf_to_ltf(prices, df_1d, ema_slope)
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position
    
    # Warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(ema_slope_12h[i]) or np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h[i]
        
        # Volume filter: current volume > 1.3x daily average
        vol_filter = vol_now > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + uptrend + volume
            if price_now > donch_high_12h[i] and ema_slope_12h[i] > 0 and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + downtrend + volume
            elif price_now < donch_low_12h[i] and ema_slope_12h[i] < 0 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend turns down
            if price_now < donch_low_12h[i] or ema_slope_12h[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend turns up
            if price_now > donch_high_12h[i] or ema_slope_12h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0