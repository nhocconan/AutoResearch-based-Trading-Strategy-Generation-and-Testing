#!/usr/bin/env python3
"""
4h_DonchianBreakout_VolumeTrend_v2
Concept: 4h Donchian(20) breakout with volume confirmation and 1d trend filter.
- Long: Close > Donchian high(20) AND Volume > 1.5x 20-period volume avg AND Close > 1d EMA50
- Short: Close < Donchian low(20) AND Volume > 1.5x 20-period volume avg AND Close < 1d EMA50
- Exit: Close crosses Donchian midpoint (mean of high/low over 20 periods)
- Position sizing: 0.25
- Target: 20-50 trades/year (80-200 total over 4 years)
- Works in bull/bear: Donchian captures breakouts, volume confirms strength, 1d EMA50 filters counter-trend noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_DonchianBreakout_VolumeTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h: Donchian channels (20-period high/low) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily: EMA50 trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian
    
    for i in range(start_idx, n):
        # Get values
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        dc_mid = donchian_mid[i]
        vol_ma = vol_ma20[i]
        ema50 = ema50_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(dc_high) or np.isnan(dc_low) or np.isnan(dc_mid) or 
            np.isnan(vol_ma) or np.isnan(ema50)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume and 1d uptrend
            if close[i] > dc_high and volume[i] > 1.5 * vol_ma and close[i] > ema50:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume and 1d downtrend
            elif close[i] < dc_low and volume[i] > 1.5 * vol_ma and close[i] < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian midpoint
            if close[i] < dc_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above Donchian midpoint
            if close[i] > dc_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals