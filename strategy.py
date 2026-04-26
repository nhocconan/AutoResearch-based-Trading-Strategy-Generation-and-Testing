#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Donchian(20) breakouts with 12h EMA50 trend filter and volume spike confirmation.
Uses ATR trailing stop (2.0x) and requires price >1.5% from EMA50 to avoid chop. Position size 0.25.
Designed for stable performance in both bull and bear markets via confluence: breakout + HTF trend + volume spike.
Tight entry conditions target ~100-150 total trades over 4 years to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels from previous 4h bar (using 4h data)
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x median volume (balanced for frequency)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR for stop (14-period on 4h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 12h EMA (50), Donchian (20), volume median (30), 4h ATR (14)
    start_idx = max(50, 20, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_median[i]) or 
            np.isnan(atr_14[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        donchian_upper_val = donchian_upper[i]
        donchian_lower_val = donchian_lower[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above Donchian upper, uptrend (close > EMA50), volume spike, price >1.5% from EMA
            long_signal = (high_val > donchian_upper_val) and \
                          (close_val > ema_50_12h_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (np.abs((close_val - ema_50_12h_val) / ema_50_12h_val * 100) > 1.5)
            # Short: break below Donchian lower, downtrend (close < EMA50), volume spike, price >1.5% from EMA
            short_signal = (low_val < donchian_lower_val) and \
                           (close_val < ema_50_12h_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (np.abs((close_val - ema_50_12h_val) / ema_50_12h_val * 100) > 1.5)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.0 * atr_14_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.0 * atr_14_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_50_12h_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_50_12h_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0