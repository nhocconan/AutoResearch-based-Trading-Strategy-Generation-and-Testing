#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and 1d volume confirmation.
# Long when price breaks above Donchian upper AND price > weekly pivot (bullish bias) AND volume > 1.2x 20-period 1d average.
# Short when price breaks below Donchian lower AND price < weekly pivot (bearish bias) AND volume > 1.2x 20-period 1d average.
# Exit when price crosses Donchian midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture breakouts aligned with weekly structure.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # === 1d Indicators: Volume Confirmation (volume > 1.2x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # === 1w Indicators: Weekly Pivot (directional bias) ===
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot = (weekly high + weekly low + weekly close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods for Donchian/volume MA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(volume_spike[i]) or np.isnan(weekly_pivot_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian midpoint
            if price < donchian_mid[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian midpoint
            if price > donchian_mid[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND price > weekly pivot (bullish bias) AND volume spike
            if price > donchian_upper[i] and price > weekly_pivot_val and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND price < weekly pivot (bearish bias) AND volume spike
            elif price < donchian_lower[i] and price < weekly_pivot_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0