#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike (>2.0x average) captures strong breakouts with low false signals. Uses discrete sizing (0.25) and ATR-based stoploss (signal→0 when price retraces 2.0x ATR from extreme). Designed for 4h timeframe to balance trade frequency and edge in both bull and bear markets via trend filter. Target 20-40 trades/year to minimize fee drag.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels from 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (20-period SMA = ~6.67h)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_high = 0.0  # track highest high since entry for longs
    entry_low = 0.0   # track lowest low since entry for shorts
    base_size = 0.25
    
    # Warmup: max of Donchian(20), EMA(50), volume(20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_12h_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower) or np.isnan(atr_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long: price CLOSES above upper Donchian with 12h uptrend and volume
        long_condition = (close_val > upper) and (close_val > ema_val) and volume_confirmed
        # Short: price CLOSES below lower Donchian with 12h downtrend and volume
        short_condition = (close_val < lower) and (close_val < ema_val) and volume_confirmed
        
        # Update extreme prices for trailing stop logic
        if position == 1:
            entry_high = max(entry_high, high_val)
        elif position == -1:
            entry_low = min(entry_low, low_val)
        
        # Stoploss: price retraces 2.0x ATR from extreme
        long_stop = (position == 1 and close_val <= entry_high - 2.0 * atr_val)
        short_stop = (position == -1 and close_val >= entry_low + 2.0 * atr_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            entry_high = high_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            entry_low = low_val
        elif long_stop:
            signals[i] = 0.0
            position = 0
            entry_high = 0.0
        elif short_stop:
            signals[i] = 0.0
            position = 0
            entry_low = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0