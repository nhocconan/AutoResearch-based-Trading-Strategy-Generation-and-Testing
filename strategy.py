#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ADXTrend
Hypothesis: Donchian(20) breakouts on 4h with volume spike and ADX(14)>25 trend filter. Targets 15-30 trades/year by requiring confluence of volume confirmation and strong trend (ADX>25). Uses discrete position sizing (0.25) to minimize fee churn. Works in bull/bear via ADX trend filter to avoid whipsaws in ranging markets.
Primary timeframe: 4h, HTF: none needed (all indicators calculated on 4h).
"""

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: volume > 2.0x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # ADX(14) for trend strength
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = high[0] - low[0]  # first TR
    
    # Calculate Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR and DM
    tr_period = 14
    atr = pd.Series(tr).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=tr_period, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr == 0, 1e-10, atr)
    di_minus = 100 * dm_minus_smooth / np.where(atr == 0, 1e-10, atr)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1e-10, (di_plus + di_minus))
    adx = pd.Series(dx).rolling(window=tr_period, min_periods=tr_period).mean().values
    
    # Trend filter: ADX > 25 indicates strong trend
    strong_trend = adx > 25
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 20 for Donchian, 20 for volume median, 14*2 for ADX
    start_idx = max(20, 20, 28)  # 28 for ADX (14 for smoothing + 14 for ADX)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_median_20[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_spike = volume_spike[i]
        trend_ok = strong_trend[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Long: price breaks above Donchian high with volume spike and strong trend
            long_entry = (close_val > highest_high[i]) and vol_spike and trend_ok
            # Short: price breaks below Donchian low with volume spike and strong trend
            short_entry = (close_val < lowest_low[i]) and vol_spike and trend_ok
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend weakness or price re-enters Donchian channel
            if not trend_ok or close_val < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend weakness or price re-enters Donchian channel
            if not trend_ok or close_val > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0