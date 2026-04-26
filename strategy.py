#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 4h Donchian(20) breakout aligned with 1d EMA50 trend and volume spike (>1.5x average) captures institutional moves with controlled frequency. Uses ATR(14) stoploss to manage risk. Designed for 4h to target 20-50 trades/year with discrete sizing (0.25). Works in bull/bear via 1d trend filter.
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss and volume spike threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 24 periods)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of Donchian(20), EMA(50), volume(24), ATR(14)
    start_idx = max(20, 50, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val) or 
            np.isnan(upper_channel) or np.isnan(lower_channel)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above upper Donchian with 1d uptrend and volume
        long_condition = (close_val > upper_channel) and uptrend and volume_confirmed
        # Short: price CLOSES below lower Donchian with 1d downtrend and volume
        short_condition = (close_val < lower_channel) and downtrend and volume_confirmed
        
        # Stoploss: ATR-based (2x ATR from entry)
        long_stop = (position == 1 and close_val <= entry_price - 2.0 * atr_val)
        short_stop = (position == -1 and close_val >= entry_price + 2.0 * atr_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_stop:
            signals[i] = 0.0
            position = 0
        elif short_stop:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0