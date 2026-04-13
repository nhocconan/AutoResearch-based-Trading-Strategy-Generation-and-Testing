#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
    # Long: 6h price breaks above Donchian(20) high AND 1d price > weekly pivot point AND 6h volume > 1.5 * 20-period avg volume
    # Short: 6h price breaks below Donchian(20) low AND 1d price < weekly pivot point AND 6h volume > 1.5 * 20-period avg volume
    # Exit: Price returns to Donchian midpoint OR volume drops below average
    # Uses 6h for breakout and volume, 1d for weekly pivot regime filter
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (~12-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian and volume (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for weekly pivot (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian Channel (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian high: highest high over 20 periods
    donch_high_6h = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Donchian low: lowest low over 20 periods
    donch_low_6h = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint
    donch_mid_6h = (donch_high_6h + donch_low_6h) / 2.0
    
    # Align 6h Donchian to 6h timeframe (no additional delay for price-based indicators)
    donch_high_aligned = align_htf_to_ltf(prices, df_6h, donch_high_6h)
    donch_low_aligned = align_htf_to_ltf(prices, df_6h, donch_low_6h)
    donch_mid_aligned = align_htf_to_ltf(prices, df_6h, donch_mid_6h)
    
    # Calculate 6h volume average (20-period)
    volume_6h = df_6h['volume'].values
    vol_avg_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_6h, vol_avg_6h)
    
    # Calculate 1d weekly pivot point (using prior week's data)
    # Weekly pivot = (prior week high + prior week low + prior week close) / 3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 5 to get prior week's OHLC (approximation for weekly data)
    # Since we don't have actual weekly data, we use 5-day lookback as proxy
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(5).values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(5).values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(5).values
    
    # Weekly pivot point
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align 1d weekly pivot to 6h (wait for completed 1d bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Regime filter: 1d price relative to weekly pivot
        price_above_pivot = close[i] > weekly_pivot_aligned[i]
        price_below_pivot = close[i] < weekly_pivot_aligned[i]
        
        # Breakout signals
        long_breakout = close[i] > donch_high_aligned[i]
        short_breakout = close[i] < donch_low_aligned[i]
        
        # Entry logic: Breakout + volume confirmation + regime filter
        long_entry = long_breakout and volume_confirm and price_above_pivot
        short_entry = short_breakout and volume_confirm and price_below_pivot
        
        # Exit logic: Return to midpoint OR volume drops below average
        long_exit = (close[i] <= donch_mid_aligned[i]) or (volume[i] < vol_avg_aligned[i])
        short_exit = (close[i] >= donch_mid_aligned[i]) or (volume[i] < vol_avg_aligned[i])
        
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

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0