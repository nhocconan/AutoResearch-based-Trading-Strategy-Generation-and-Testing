#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and 1d volume confirmation.
# Long when price breaks above 6h Donchian upper band AND weekly pivot is bullish (close > weekly pivot) AND 1d volume > 1.5x 20-period average.
# Short when price breaks below 6h Donchian lower band AND weekly pivot is bearish (close < weekly pivot) AND 1d volume > 1.5x 20-period average.
# Exit when price crosses the 6h Donchian midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture breakouts aligned with weekly structure in both bull and bear markets.
# Target: 80-160 total trades over 4 years (20-40/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Donchian(20) channels ===
    # Upper band: 20-period high
    high_series = pd.Series(high)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    low_series = pd.Series(low)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    # Middle band: average of upper and lower
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # === Weekly Indicators: Pivot point (from previous week) ===
    df_1w = get_htf_data(prices, '1w')
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    # Weekly direction: bullish if close > pivot, bearish if close < pivot
    weekly_bullish = close > weekly_pivot_aligned
    weekly_bearish = close < weekly_pivot_aligned
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for Donchian/volume MA)
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_weekly_bullish = weekly_bullish[i]
        is_weekly_bearish = weekly_bearish[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below Donchian middle
            if price < donchian_middle[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above Donchian middle
            if price > donchian_middle[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND weekly bullish AND volume spike
            if price > donchian_upper[i] and is_weekly_bullish and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND weekly bearish AND volume spike
            elif price < donchian_lower[i] and is_weekly_bearish and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_1dVolumeSpike_V1"
timeframe = "6h"
leverage = 1.0