#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-day EMA200 filter and volume confirmation.
Uses Elder Ray to measure bull/bear power relative to EMA13, enters long when bull power > 0 and price > EMA200(1d),
short when bear power < 0 and price < EMA200(1d). Volume confirmation filters low-momentum entries.
Trades only in alignment with higher timeframe trend (1-day EMA200) to avoid counter-trend whipsaws.
Targets 60-120 total trades over 4 years (15-30/year) with controlled risk via trend alignment.
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
    
    # Calculate EMA13 for Elder Ray (6-period EMA equivalent for 6h? Actually use 13)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 1d data for EMA200 trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).values
    
    # Calculate 1-day volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    
    # Align 1d indicators to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Elder Ray power aligned with 1d trend + volume
        bullish = bull_power[i] > 0 and close[i] > ema200_1d_aligned[i]
        bearish = bear_power[i] < 0 and close[i] < ema200_1d_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = bullish and vol_confirm
        short_entry = bearish and vol_confirm
        
        # Exit when Elder Ray power reverses
        exit_long = position == 1 and bull_power[i] <= 0
        exit_short = position == -1 and bear_power[i] >= 0
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_power_volume"
timeframe = "6h"
leverage = 1.0