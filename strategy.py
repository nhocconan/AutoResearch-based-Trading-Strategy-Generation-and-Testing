#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1
Hypothesis: Trade 6h Donchian(20) breakouts in direction of 12h EMA50 trend with 1d volume confirmation (2.0x median volume). 
Uses ATR trailing stop (2.5x) and only enters when price >1.5% from EMA50 to avoid chop. Position size 0.25.
Designed for low trade frequency (~15-30/year) by requiring confluence: 6h breakout + 12h trend + 1d volume spike + momentum filter.
Works in bull markets (breakouts with trend) and bear markets (short breakdowns against trend). Focus on BTC/ETH as primary targets.
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
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d median volume for spike detection
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    
    # ATR for stop (14-period on 6h)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period on 6h)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Price distance from EMA50 to avoid chop (>1.5%)
    ema_distance = np.abs((close - ema_50_12h_aligned) / ema_50_12h_aligned * 100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    # Warmup: max of 12h EMA (50), 1d volume median (30), 6h ATR (14), Donchian (20), distance calc
    start_idx = max(50, 30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_distance[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        atr_14_val = atr_14[i]
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        ema_distance_val = ema_distance[i]
        
        if position == 0:
            # Long: break above Donchian high, uptrend (close > EMA50), volume spike, price >1.5% from EMA
            long_signal = (high_val > highest_high_val) and \
                          (close_val > ema_50_12h_val) and \
                          (volume_val > 2.0 * vol_median_1d_val) and \
                          (ema_distance_val > 1.5)
            # Short: break below Donchian low, downtrend (close < EMA50), volume spike, price >1.5% from EMA
            short_signal = (low_val < lowest_low_val) and \
                           (close_val < ema_50_12h_val) and \
                           (volume_val > 2.0 * vol_median_1d_val) and \
                           (ema_distance_val > 1.5)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                long_stop = entry_price - 2.5 * atr_14_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                short_stop = entry_price + 2.5 * atr_14_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Update trailing stop: move stop up as price makes new highs
            long_stop = max(long_stop, high_val - 2.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close < EMA50)
            if (low_val < long_stop) or (close_val < ema_50_12h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Update trailing stop: move stop down as price makes new lows
            short_stop = min(short_stop, low_val + 2.5 * atr_14_val)
            # Exit: trailing stop hit or trend reversal (close > EMA50)
            if (high_val > short_stop) or (close_val > ema_50_12h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_12hTrend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0