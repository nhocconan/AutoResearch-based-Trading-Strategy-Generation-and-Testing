#!/usr/bin/env python3
"""
4h_Donchian20_1dVolume_TrendFilter_v2
Concept: 4h Donchian(20) breakout with daily volume spike and trend filter.
- Long: Price > Donchian high(20) AND daily volume > 1.5x 20-period average AND close > EMA(50)
- Short: Price < Donchian low(20) AND daily volume > 1.5x 20-period average AND close < EMA(50)
- Exit: Price crosses back through Donchian midpoint
- Position sizing: 0.25
- Target: 75-200 total trades over 4 years
- Works in bull/bear: EMA(50) trend filter adapts, volume confirms institutional interest
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dVolume_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 4h: EMA Trend Filter (50-period) ===
    close = prices['close'].values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 4h: Donchian Channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Donchian high/low using rolling window
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        ema50 = ema_50[i]
        
        # Skip if any value is NaN
        if (np.isnan(dh) or np.isnan(dl) or np.isnan(dm) or np.isnan(ema50)):
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
            if close[i] > dh and close[i] > ema50 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with downtrend and volume spike
            elif close[i] < dl and close[i] < ema50 and vol_condition:
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