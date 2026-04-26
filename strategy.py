#!/usr/bin/env python3
"""
6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: 6h Donchian breakout with weekly trend filter and 1d volume spike confirmation.
- Uses weekly Donchian(20) for breakout direction (structure from higher timeframe)
- 1d EMA50 filter ensures trades align with daily trend (bull/bear agnostic)
- 1d volume spike (>1.5x 20-period average) confirms breakout strength
- Long when price breaks above weekly Donchian high AND close > 1d EMA50 AND volume spike
- Short when price breaks below weekly Donchian low AND close < 1d EMA50 AND volume spike
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the daily trend and using weekly structure for breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Load daily data ONCE before loop for trend filter and volume spike
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate daily volume spike (>1.5x 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_spike = volume_1d > (vol_ma20_1d * 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]
        breakout_down = close[i] < donchian_low_aligned[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_confirmed = volume_spike_aligned[i] > 0.5  # Boolean as float
        
        if position == 0:
            # Long: breakout above weekly Donchian high AND daily uptrend AND volume spike
            if breakout_up and daily_uptrend and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly Donchian low AND daily downtrend AND volume spike
            elif breakout_down and daily_downtrend and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below weekly Donchian low OR daily trend turns down
            if close[i] < donchian_low_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above weekly Donchian high OR daily trend turns up
            if close[i] > donchian_high_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyDonchianBreakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0