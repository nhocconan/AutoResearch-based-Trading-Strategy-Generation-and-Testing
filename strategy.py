#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility regime filter and volume confirmation.
- Long: price breaks above Donchian upper (20) + ATR(14)/ATR(50) > 0.8 (normal/high vol regime) + volume > 1.5x 20-period avg
- Short: price breaks below Donchian lower (20) + ATR(14)/ATR(50) > 0.8 + volume > 1.5x 20-period avg
- Exit: trailing stop (2.5x ATR from extreme) OR Donchian breakout in opposite direction
- Uses ATR regime filter to avoid low-volatility choppy markets where breakouts fail
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: volatility regime adapts to market conditions
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
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
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period) on primary timeframe
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 1d data ONCE before loop for ATR regime (using same ATR calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) and ATR(50) on 1d for regime filter
    tr1_1d = np.abs(df_1d['high'].values[1:] - df_1d['low'].values[1:])
    tr2_1d = np.abs(df_1d['high'].values[1:] - df_1d['close'].values[:-1])
    tr3_1d = np.abs(df_1d['low'].values[1:] - df_1d['close'].values[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with close
    atr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    
    # ATR regime: ratio of short-term to long-term ATR (> 0.8 = normal/high volatility)
    atr_ratio_1d = atr14_1d / atr50_1d
    
    # Align HTF indicators to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50, 20)  # Need 20 for Donchian/volume, 14/50 for ATR, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(atr50[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs channel)
        breakout_up = close[i] > donchian_upper[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Volatility regime filter: ATR ratio > 0.8 (avoid low-volatility chop)
        vol_regime = atr_ratio_aligned[i] > 0.8
        
        if position == 0:
            # Long: Donchian upper breakout + volume spike + volatility regime
            if breakout_up and volume_spike and vol_regime:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian lower breakout + volume spike + volatility regime
            elif breakout_down and volume_spike and vol_regime:
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
            breakout_down_exit = close[i] < donchian_lower[i]
            
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
            breakout_up_exit = close[i] > donchian_upper[i]
            
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