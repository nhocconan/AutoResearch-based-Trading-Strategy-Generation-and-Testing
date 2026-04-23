#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: price breaks above Donchian upper (20) + ATR(14)/ATR(50) < 0.8 (low volatility regime) + volume > 1.5x 20-period avg volume
- Short: price breaks below Donchian lower (20) + ATR(14)/ATR(50) < 0.8 + volume > 1.5x 20-period avg volume
- Exit: trailing stop (2.5x ATR from extreme) OR Donchian breakout in opposite direction
- Uses ATR ratio regime filter to avoid whipsaws in high volatility (bear markets) and capture breakouts in low vol
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk
- Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) and ATR(50) for regime filter
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime filter: low volatility when ATR(14)/ATR(50) < 0.8
    atr_ratio = np.where(atr50 > 0, atr14 / atr50, 1.0)
    low_vol_regime = atr_ratio < 0.8
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for Donchian channels (20-period)
    df_1d = get_htf_data(prices, '1d')
    donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need 50 for ATR50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(atr50[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_upper_aligned[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower_aligned[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 1.5x average) and low volatility regime
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        regime_ok = low_vol_regime[i]
        
        if position == 0:
            # Long: Donchian breakout up + low vol regime + volume spike
            if breakout_up and regime_ok and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + low vol regime + volume spike
            elif breakout_down and regime_ok and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Donchian breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr14[i]
            breakout_down_exit = close[i] < donchian_lower_aligned[i]
            
            if trailing_stop_long or breakout_down_exit:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, low[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from short extreme (trailing stop)
            # 2. Donchian breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr14[i]
            breakout_up_exit = close[i] > donchian_upper_aligned[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATRRegime_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0