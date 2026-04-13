#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d regime filter (ADX) and volume confirmation.
    # Elder Ray measures bull/bear power relative to EMA13. ADX > 25 indicates trending market.
    # In trending markets (ADX>25), go long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
    # In ranging markets (ADX<20), fade extremes: short when Bull Power > 0.8*std, long when Bear Power < -0.8*std.
    # Volume filter ensures participation. Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13
    bear_power = df_1d['low'].values - ema13
    
    # Calculate 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, (dm_plus_smooth / atr) * 100, 0)
    di_minus = np.where(atr > 0, (dm_minus_smooth / atr) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 20
        
        # Calculate rolling std for power extremes (using last 20 periods)
        if i >= 20:
            bull_std = np.nanstd(bull_power_aligned[i-20:i])
            bear_std = np.nanstd(bear_power_aligned[i-20:i])
        else:
            bull_std = bear_std = 1.0  # Avoid division by zero early on
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if is_trending and volume_filter:
            # In trending market: go with the power
            # Long when Bull Power > 0 and rising (current > previous)
            # Short when Bear Power < 0 and falling (current < previous)
            if i > 0:
                bull_rising = bull_power_aligned[i] > bull_power_aligned[i-1]
                bear_falling = bear_power_aligned[i] < bear_power_aligned[i-1]
                long_entry = bull_power_aligned[i] > 0 and bull_rising
                short_entry = bear_power_aligned[i] < 0 and bear_falling
        elif is_ranging and volume_filter:
            # In ranging market: fade extremes
            long_entry = bear_power_aligned[i] < -0.8 * bear_std
            short_entry = bull_power_aligned[i] > 0.8 * bull_std
        
        # Exit conditions: opposite power signal or regime change
        long_exit = False
        short_exit = False
        
        if is_trending:
            # Exit long when Bull Power turns negative
            # Exit short when Bear Power turns positive
            long_exit = bull_power_aligned[i] < 0
            short_exit = bear_power_aligned[i] > 0
        else:  # ranging
            # Exit when power returns to neutral zone
            long_exit = bear_power_aligned[i] > -0.2 * bear_std
            short_exit = bull_power_aligned[i] < 0.2 * bull_std
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0