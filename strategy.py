#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 12h/1d Multi-timeframe Price Channel Breakout with Volume Confirmation
# Uses 12h Donchian breakout direction + 4h entry timing + 1d volume filter
# 12h timeframe provides trend direction (fewer signals), 4h provides precise entry
# Volume filter ensures breakouts have institutional participation
# Works in bull/bear by following higher timeframe trend direction
# Target: 80-150 total trades over 4 years (20-38/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === Multi-timeframe Data Loading (ONCE before loop) ===
    # 12h for trend direction
    df_12h = get_htf_data(prices, '12h')
    # 1d for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # === 12h Donchian Channel (Trend Direction) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align 12h Donchian levels to 4h
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    
    # === 1d Volume Filter ===
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / np.where(avg_vol_1d > 0, avg_vol_1d, 1)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 4h Price Data ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Volume Spike (Entry Timing) ===
    vol_series = pd.Series(volume)
    avg_vol_4h = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for 20-period calculations with buffer
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_12h_aligned[i]) or np.isnan(donchian_low_12h_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(avg_vol_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Determine trend direction from 12h Donchian breakout
        trend_up = price > donchian_high_12h_aligned[i]
        trend_down = price < donchian_low_12h_aligned[i]
        
        # Volume conditions: both 1d and 4h volume must be elevated
        vol_1d_high = vol_ratio_1d_aligned[i] > 1.5  # 1d volume > 1.5x average
        vol_4h_high = vol > 2.0 * avg_vol_4h[i]      # 4h volume > 2x average
        
        if position == 0:
            # Long: 12h uptrend + volume confirmation on both timeframes
            if trend_up and vol_1d_high and vol_4h_high:
                position = 1
                signals[i] = position_size
            # Short: 12h downtrend + volume confirmation on both timeframes
            elif trend_down and vol_1d_high and vol_4h_high:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 12h trend reverses or volume dries up
            if not trend_up or not (vol_1d_high and vol_4h_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: 12h trend reverses or volume dries up
            if not trend_down or not (vol_1d_high and vol_4h_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_1d_MTF_Volume_Breakout"
timeframe = "4h"
leverage = 1.0