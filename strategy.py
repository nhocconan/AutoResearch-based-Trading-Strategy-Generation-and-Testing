#!/usr/bin/env python3
"""
4h_PriceChannel_VolumeBreakout_v1
Concept: 4h price channel breakout with daily volume confirmation and multi-timeframe trend filter.
- Long: Price > 4h Donchian high(20) AND daily volume > 1.5x 20-period avg AND 1d EMA(50) > 1d EMA(200)
- Short: Price < 4h Donchian low(20) AND daily volume > 1.5x 20-period avg AND 1d EMA(50) < 1d EMA(200)
- Exit: Price crosses back through 4h Donchian midpoint
- Position sizing: 0.25
- Target: 75-200 total trades over 4 years
- Works in bull/bear: 1d EMA trend filter adapts, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_PriceChannel_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === Daily: EMA Trend Filter (50 and 200) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian high/low using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        ema50 = ema_50_1d_aligned[i]
        ema200 = ema_200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or 
            np.isnan(ema50) or np.isnan(ema200)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current daily volume > 1.5x 20-period average
        vol_1d_vals = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_vals)
        current_vol = vol_1d_aligned[i]
        vol_condition = current_vol > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with uptrend and volume spike
            if close[i] > dh and ema50 > ema200 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume spike
            elif close[i] < dl and ema50 < ema200 and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if close[i] < dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if close[i] > dm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals