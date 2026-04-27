#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation.
# Long when price breaks above weekly Donchian high (20-week) with volume > 1.5x average.
# Short when price breaks below weekly Donchian low (20-week) with volume > 1.5x average.
# Exit when price returns to weekly mid-point (average of high and low).
# Uses weekly timeframe for trend and daily for execution to reduce whipsaw and capture major trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = np.full(len(high_weekly), np.nan)
    donchian_low = np.full(len(low_weekly), np.nan)
    
    for i in range(19, len(high_weekly)):
        donchian_high[i] = np.max(high_weekly[i-19:i+1])
        donchian_low[i] = np.min(low_weekly[i-19:i+1])
    
    # Weekly mid-point for exit
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly indicators to daily timeframe
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_daily = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Volume moving average (20-period) for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Start after warmup periods
    start_idx = max(19, 19)  # weekly lookback and volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_daily[i]) or np.isnan(donchian_low_daily[i]) or 
            np.isnan(donchian_mid_daily[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require above-average volume
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above weekly Donchian high with volume
            if price > donchian_high_daily[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below weekly Donchian low with volume
            elif price < donchian_low_daily[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly mid-point
            if price < donchian_mid_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to weekly mid-point
            if price > donchian_mid_daily[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchianBreakout_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0