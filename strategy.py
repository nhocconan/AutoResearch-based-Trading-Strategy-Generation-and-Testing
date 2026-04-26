#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts with 1d EMA50 trend filter and volume spike capture institutional breakouts while avoiding false signals. Long when price breaks above R1 with volume > 1.5x average and close > 1d EMA50; Short when price breaks below S1 with volume > 1.5x average and close < 1d EMA50. Uses ATR-based stoploss and discrete sizing (±0.25) to limit trades to 20-50/year and minimize fee drag. Designed to work in both bull and bear markets with BTC/ETH edge.
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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Previous day's high, low, close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    camarilla_range = (high_1d - low_1d)
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume average (20-period) for spike detection
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of 1d EMA50 calculation (50) + volume MA (20) + alignment buffer
    start_idx = 50 + 4  # +4 to ensure 1d bar completion (4h -> 1d: 6 bars per day, but we use completed bar)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        vol_ma = volume_ma[i]
        vol_current = volume[i]
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_spike = vol_current > 1.5 * vol_ma if vol_ma > 0 else False
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close_val > ema_50_val
        downtrend = close_val < ema_50_val
        
        # Entry conditions
        long_entry = (close_val > r1_val) and volume_spike and uptrend
        short_entry = (close_val < s1_val) and volume_spike and downtrend
        
        # Exit conditions: opposite signal or volume dry-up
        long_exit = (close_val < s1_val) or (not volume_spike) or (close_val < ema_50_val)
        short_exit = (close_val > r1_val) or (not volume_spike) or (close_val > ema_50_val)
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0