#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrendFilter_WeeklyVolume_v1
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and weekly volume confirmation.
- Primary timeframe: 12h for low trade frequency (target: 50-150 total trades over 4 years)
- Entry: Price breaks above/below 20-period Donchian channel + 1d EMA50 trend alignment + weekly volume > 1.5x 20-period average
- Exit: Price returns to mid-channel (mean of upper/lower Donchian bands)
- Uses discrete position sizing (0.25) to minimize fee churn
- Works in bull/bear markets by trading with the 1d trend and using Donchian breakouts for momentum
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Load weekly data ONCE before loop for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly volume 20-period average for confirmation
    volume_1w = df_1w['volume'].values
    vol_ma20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma20_1w)
    
    # Donchian channel calculations (20 periods)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = period20_high
    donchian_lower = period20_low
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma20_1w_aligned[i]) or
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(donchian_mid[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        price_above_upper = close[i] > donchian_upper[i]
        price_below_lower = close[i] < donchian_lower[i]
        price_at_mid = abs(close[i] - donchian_mid[i]) < (donchian_upper[i] - donchian_lower[i]) * 0.1
        
        # 1d trend filter
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Weekly volume confirmation (volume > 1.5x 20-period average)
        volume_confirm = volume[i] > 1.5 * vol_ma20_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND uptrend AND volume confirmation
            if price_above_upper and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND downtrend AND volume confirmation
            elif price_below_lower and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price returns to mid-channel (within 10% of mid)
            if price_at_mid:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price returns to mid-channel (within 10% of mid)
            if price_at_mid:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrendFilter_WeeklyVolume_v1"
timeframe = "12h"
leverage = 1.0