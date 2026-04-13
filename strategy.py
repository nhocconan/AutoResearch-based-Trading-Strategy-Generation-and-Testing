#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian breakout with daily volume confirmation and weekly trend filter.
# Donchian channels identify breakouts with clear support/resistance levels.
# Volume confirmation ensures breakouts have institutional participation.
# Weekly trend filter avoids counter-trend trades in strong trends.
# Target: 15-30 trades per year (60-120 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate average daily volume (20-period) for volume confirmation
    daily_vol = df_1d['volume'].values
    avg_daily_vol = np.zeros(len(daily_vol))
    for i in range(20, len(daily_vol)):
        avg_daily_vol[i] = np.mean(daily_vol[i-20:i])
    
    avg_daily_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_vol)
    
    # Calculate weekly EMA trend filter (21-period)
    weekly_close = df_1w['close'].values
    ema_weekly = np.zeros(len(weekly_close))
    if len(weekly_close) > 0:
        ema_multiplier = 2 / (21 + 1)
        ema_weekly[0] = weekly_close[0]
        for i in range(1, len(weekly_close)):
            ema_weekly[i] = (weekly_close[i] - ema_weekly[i-1]) * ema_multiplier + ema_weekly[i-1]
    
    ema_weekly_aligned = align_htf_to_ltf(prices, df_1w, ema_weekly)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_daily_vol_aligned[i]) or np.isnan(ema_weekly_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        daily_vol_avg = avg_daily_vol_aligned[i]
        weekly_ema = ema_weekly_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average daily volume
        volume_confirm = vol > 1.5 * daily_vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high with volume + above weekly EMA
            if price > donchian_high[i] and volume_confirm and price > weekly_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume + below weekly EMA
            elif price < donchian_low[i] and volume_confirm and price < weekly_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or breaks below low
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price < midpoint or price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or breaks above high
            midpoint = (donchian_high[i] + donchian_low[i]) / 2
            if price > midpoint or price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_1w_Donchian_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0