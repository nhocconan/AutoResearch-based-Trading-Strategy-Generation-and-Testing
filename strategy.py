#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + 12h ADX Trend + Volume Spike
Hypothesis: Donchian channel breakouts capture strong momentum. 12h ADX > 25 filters for trending regimes, 
while volume spike confirms institutional participation. 6h timeframe targets 12-37 trades/year, minimizing 
fee drag. Works in bull/bear by trading only when ADX confirms trend strength.
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Donchian channels (20-period) on 6h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ADX(14) for trend strength
    # ADX requires +DI, -DI, and DX calculation
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).sub(pd.Series(low_12h))
    tr2 = pd.Series(high_12h).sub(pd.Series(close_12h).shift(1)).abs()
    tr3 = pd.Series(low_12h).sub(pd.Series(close_12h).shift(1)).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM and -DM
    up_move = pd.Series(high_12h).diff()
    down_move = pd.Series(low_12h).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_12h = pd.Series(tr_12h.values).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h * 100
    minus_di_12h = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h * 100
    
    # DX and ADX
    dx = np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h) * 100
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 12h ADX to 6h timeframe (completed 12h bar only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (strict)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 14*3) + 1  # Donchian20 + ADX calculation + 1 for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(adx_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper band AND strong trend AND volume spike
            long_entry = (curr_high > high_20[i]) and strong_trend and vol_spike
            # Short: price breaks below Donchian lower band AND strong trend AND volume spike
            short_entry = (curr_low < low_20[i]) and strong_trend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price falls below Donchian lower band OR loss of trend (ADX < 20)
            if (curr_low < low_20[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price rises above Donchian upper band OR loss of trend (ADX < 20)
            if (curr_high > high_20[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_12hADX_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0