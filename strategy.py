#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Donchian(20) breakouts with 4h EMA50 trend filter and volume spike confirmation.
Uses ATR trailing stop (2.0x) and requires price >1.0% from EMA50 to avoid chop. Position size 0.25.
Designed for stable performance in both bull and bear markets via confluence: price channel breakout + HTF trend + volume spike.
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
    
    # Get 4h data for HTF trend filter (same timeframe as primary)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels from previous 4h bar
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
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
    
    # Warmup: max of 4h EMA (50), Donchian (20), volume median (30), 4h ATR (14)
    start_idx = max(50, 20, 30, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
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
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_14_val = atr_14[i]
        
        if position == 0:
            # Long: break above Donchian high, uptrend (close > EMA50), volume spike, price >1.0% from EMA
            long_signal = (high_val > donchian_high_val) and \
                          (close_val > ema_50_4h_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (np.abs((close_val - ema_50_4h_val) / ema_50_4h_val * 100) > 1.0)
            # Short: break below Donchian low, downtrend (close < EMA50), volume spike, price >1.0% from EMA
            short_signal = (low_val < donchian_low_val) and \
                           (close_val < ema_50_4h_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (np.abs((close_val - ema_50_4h_val) / ema_50_4h_val * 100) > 1.0)
            
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
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_50_4h_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.0 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_50_4h_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0