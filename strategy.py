#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrendFilter_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation. Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets. Uses discrete position sizing (0.25) to reduce churn. Works in bull markets via breakout continuation and in bear markets via breakdown continuation with trend filter alignment.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h Donchian(20) - using 20 periods of 6h data = ~5 days
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), Donchian (20), volume MA (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        high_val = high[i]
        low_val = low[i]
        close_val = close[i]
        hh_val = highest_high[i]
        ll_val = lowest_low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation and uptrend
            long_signal = (high_val > hh_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below Donchian low with volume confirmation and downtrend
            short_signal = (low_val < ll_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal or Donchian midpoint reversion
            if (close_val < ema_50_1d_val or 
                close_val < (hh_val + ll_val) / 2):  # exit at midpoint
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or Donchian midpoint reversion
            if (close_val > ema_50_1d_val or 
                close_val > (hh_val + ll_val) / 2):  # exit at midpoint
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dTrendFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0