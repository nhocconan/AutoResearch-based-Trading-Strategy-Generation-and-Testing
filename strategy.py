#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_v1
Hypothesis: Trade daily Donchian(20) breakouts with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high AND weekly trend is up (price > weekly EMA34).
Short when price breaks below 20-day low AND weekly trend is down (price < weekly EMA34).
Use volume spike (volume > 1.5 * 20-day ATR) to confirm breakouts.
Only trade in direction of weekly trend to avoid counter-trend whipsaws.
Target: 15-25 trades/year to minimize fee drag while capturing sustained trends.
Discrete sizing: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 20-day Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day ATR for volume confirmation
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for Donchian (20) and ATR (20)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_20[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 1.5 * 20-day ATR
        volume_spike = volume[i] > 1.5 * atr_20[i]
        
        # Determine weekly trend regime
        # Bull regime: price > weekly EMA34
        # Bear regime: price < weekly EMA34
        if close[i] > ema_34_1w_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1w_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'neutral'  # no trades (rare)
        
        if position == 0:
            # Long setup: price breaks above 20-day high AND volume spike AND bull regime
            long_setup = (high[i] > high_20[i-1]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below 20-day low AND volume spike AND bear regime
            short_setup = (low[i] < low_20[i-1]) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price breaks below 20-day low OR regime turns bearish OR max holding period (60 bars = 2 months)
            if (low[i] < low_20[i]) or (regime == 'bear') or (bars_since_entry >= 60):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price breaks above 20-day high OR regime turns bullish OR max holding period (60 bars = 2 months)
            if (high[i] > high_20[i]) or (regime == 'bull') or (bars_since_entry >= 60):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "1d_WeeklyDonchianBreakout_v1"
timeframe = "1d"
leverage = 1.0