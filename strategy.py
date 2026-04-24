#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for pivot direction (price above/below weekly pivot point).
- Entry: Long when price breaks above Donchian upper (20) AND weekly pivot bullish AND volume > 1.5x 20-period average.
         Short when price breaks below Donchian lower (20) AND weekly pivot bearish AND volume > 1.5x 20-period average.
- Exit: Opposite Donchian breakout OR price crosses weekly pivot in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels identify breakouts with clear structure.
- Weekly pivot provides higher-timeframe bias to avoid counter-trend trades.
- Volume confirmation reduces false breakouts.
- Works in bull markets (buy breakouts with weekly bias up) and bear markets (sell breakdowns with weekly bias down).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly pivot point calculation (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly pivot: (Prior week High + Low + Close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    
    # Align weekly pivot to 6h timeframe (wait for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot, additional_delay_bars=0)
    
    # Weekly pivot bias: bullish if price > pivot, bearish if price < pivot
    weekly_bullish = close > weekly_pivot_aligned
    weekly_bearish = close < weekly_pivot_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need sufficient data for Donchian/volume
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(avg_vol_20[i]) or np.isnan(weekly_pivot_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses weekly pivot in opposite direction
        if position != 0:
            # Exit long: price breaks below Donchian lower OR price falls below weekly pivot
            if position == 1:
                if curr_close < donchian_lower[i] or curr_close < weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian upper OR price rises above weekly pivot
            elif position == -1:
                if curr_close > donchian_upper[i] or curr_close > weekly_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with weekly pivot bias and volume confirmation
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-period average
            volume_confirmed = curr_volume > 1.5 * avg_vol_20[i]
            
            # Long: price breaks above Donchian upper AND weekly pivot bullish AND volume confirmed
            if curr_close > donchian_upper[i] and weekly_bullish[i] and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND weekly pivot bearish AND volume confirmed
            elif curr_close < donchian_lower[i] and weekly_bearish[i] and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivot_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0