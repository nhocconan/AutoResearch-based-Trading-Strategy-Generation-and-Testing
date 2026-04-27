#!/usr/bin/env python3
"""
4h_Camarilla_R2_S2_Breakout_1dTrend_Volume_Spike
Hypothesis: Use price closing beyond stronger Camarilla R2/S2 levels (more significant than R1/S1) combined with volume spike and daily EMA34 trend filter. R2/S2 breakouts indicate stronger momentum and fewer false signals. Target 15-25 trades/year to avoid fee drag. Works in both bull (breakouts continue) and bear (false breakdowns reversed quickly).
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
    
    # Get daily data for Camarilla pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    
    # Camarilla R2 and S2 (stronger breakout levels)
    r2 = typical_price + (range_ * 1.1 / 12)
    s2 = typical_price - (range_ * 1.1 / 12)
    
    # Align levels to 4h timeframe (use previous day's levels)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # Volume confirmation: volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for volume MA and EMA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        r2_level = r2_aligned[i]
        s2_level = s2_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above R2 + volume spike + uptrend (price > EMA34)
            if close[i] > r2_level and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below S2 + volume spike + downtrend (price < EMA34)
            elif close[i] < s2_level and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S2 or trend turns down
            if close[i] < s2_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above R2 or trend turns up
            if close[i] > r2_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R2_S2_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0