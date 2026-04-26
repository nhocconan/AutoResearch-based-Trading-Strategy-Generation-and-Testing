#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts aligned with 1d EMA50 trend, confirmed by volume spikes (>2x median). Uses ATR trailing stop (2.5x) and discrete position sizing (0.25). Designed for low-frequency, high-conviction trades in both bull and bear markets by requiring confluence of price structure (breakout), HTF trend, and momentum (volume). Targets 50-150 total trades over 4 years.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 2.0x median volume (50-period)
    vol_median = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    # ATR for stop (20-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1d EMA (50), Donchian (20), volume median (50), ATR (20)
    start_idx = max(50, 20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_median[i]) or
            np.isnan(atr_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_20_val = atr_20[i]
        
        if position == 0:
            # Long: break above Donchian high, uptrend (close > EMA50), volume spike
            long_signal = (high_val > donchian_high_val) and \
                          (close_val > ema_50_1d_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: break below Donchian low, downtrend (close < EMA50), volume spike
            short_signal = (low_val < donchian_low_val) and \
                           (close_val < ema_50_1d_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_20_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_20_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((low_val < long_stop) or (close_val < ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_20_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 4 and ((high_val > short_stop) or (close_val > ema_50_1d_val)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0