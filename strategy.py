#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Donchian(20) breakout with 1d EMA50 trend filter and volume spike (>2.0x average) captures strong breakouts with low false signals. Uses discrete sizing (0.30) and ATR-based stoploss (signal→0 when price < highest - 2.0*ATR). Designed for 4h timeframe to balance trade frequency and edge. Target 20-50 trades/year to minimize fee drag. Works in bull via trend filter and in bear via short signals from downtrend breaks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    base_size = 0.30
    
    # Warmup: max of Donchian(20), EMA(50), volume(20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
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
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long: price CLOSES above upper Donchian with 1d uptrend and volume
        long_condition = (close_val > upper_channel) and (close_val > ema_val) and volume_confirmed
        # Short: price CLOSES below lower Donchian with 1d downtrend and volume
        short_condition = (close_val < lower_channel) and (close_val < ema_val) and volume_confirmed
        
        # Update trailing extremes
        if position == 1:
            highest_since_entry = max(highest_since_entry, high_val)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low_val)
        
        # Stoploss: ATR-based trailing stop
        long_stop = (position == 1 and highest_since_entry > 0 and 
                     close_val < highest_since_entry - 2.0 * atr_val)
        short_stop = (position == -1 and lowest_since_entry > 0 and 
                      close_val > lowest_since_entry + 2.0 * atr_val)
        
        # Exit: price retests broken level (mean reversion touch)
        long_exit = (position == 1 and close_val <= upper_channel)
        short_exit = (position == -1 and close_val >= lower_channel)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
            highest_since_entry = high_val
            lowest_since_entry = 0.0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
            highest_since_entry = 0.0
            lowest_since_entry = low_val
        elif long_stop or long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        elif short_stop or short_exit:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0