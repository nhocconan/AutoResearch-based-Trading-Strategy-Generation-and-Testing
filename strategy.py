#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour Donchian breakout with volume and ADX filter for trend confirmation.
# Long when price breaks above 4h Donchian upper (20), 4h ADX > 25, volume > 1.5x 20-period avg.
# Short when price breaks below 4h Donchian lower (20), 4h ADX > 25, volume > 1.5x 20-period avg.
# Exit when price crosses back inside the Donchian channel.
# Uses 4h for trend direction and structure, 1h for precise entry timing.
# Volume and ADX filters reduce false breakouts. Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_Donchian_20_4hADX25_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full_like(high_4h, np.nan, dtype=float)
    donchian_low = np.full_like(low_4h, np.nan, dtype=float)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 4h ADX calculation (14-period)
    # Calculate +DM, -DM, TR
    plus_dm = np.zeros_like(high_4h)
    minus_dm = np.zeros_like(high_4h)
    tr = np.zeros_like(high_4h)
    
    for i in range(1, len(high_4h)):
        high_diff = high_4h[i] - high_4h[i-1]
        low_diff = low_4h[i-1] - low_4h[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high_4h[i] - low_4h[i],
            abs(high_4h[i] - high_4h[i-1]),
            abs(low_4h[i] - low_4h[i-1])
        )
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nansum(data[1:period])  # Initial value
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    tr14 = wilder_smooth(tr, 14)
    plus_dm14 = wilder_smooth(plus_dm, 14)
    minus_dm14 = wilder_smooth(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    
    def wilder_smooth_adx(data, period):
        smoothed = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return smoothed
        smoothed[period-1] = np.nansum(data[1:period])  # Initial value
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + data[i]
        return smoothed
    
    adx14 = wilder_smooth_adx(dx, 14)
    
    # Align ADX to 1h timeframe
    adx14_aligned = align_htf_to_ltf(prices, df_4h, adx14)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Sufficient warmup for Donchian and ADX
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx14_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, ADX > 25, volume filter
            long_cond = (close[i] > donchian_high_aligned[i]) and (adx14_aligned[i] > 25) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, ADX > 25, volume filter
            short_cond = (close[i] < donchian_low_aligned[i]) and (adx14_aligned[i] > 25) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Donchian low
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses back above Donchian high
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals