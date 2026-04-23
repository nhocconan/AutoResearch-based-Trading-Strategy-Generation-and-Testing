#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR-based volatility filter and volume confirmation.
- Long: price breaks above Donchian upper (20-period) + ATR(1d)/ATR(20d) < 0.8 (low vol regime) + volume > 1.5x 20-period avg
- Short: price breaks below Donchian lower (20-period) + ATR(1d)/ATR(20d) < 0.8 (low vol regime) + volume > 1.5x 20-period avg
- Exit: ATR(14) trailing stop (2.5x ATR from extreme) OR Donchian breakout in opposite direction
- Uses ATR regime filter to avoid false breakouts in high volatility and catch momentum in low vol
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: volatility regime adapts to market conditions
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag on 4h timeframe
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
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR(14) and ATR(140) on 1d for volatility regime filter
    # ATR(140) ≈ 20-day ATR (since 20 trading days ≈ 1 month, but using 140 for stability)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = np.abs(high_1d[1:] - low_1d[1:])
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # align with close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_140_1d = pd.Series(tr_1d).rolling(window=140, min_periods=140).mean().values
    
    # Volatility regime: ATR(14)/ATR(140) < 0.8 indicates low volatility regime
    vol_regime = atr_14_1d / atr_140_1d < 0.8
    
    # Align HTF indicators to 4h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20, 140)  # Need 20 for Donchian/volume, 14 for ATR, 140 for ATR regime
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Low volatility regime filter
        low_vol_regime = vol_regime_aligned[i]
        
        if position == 0:
            # Long: Donchian upper breakout + low vol regime + volume spike
            if breakout_up and low_vol_regime and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian lower breakout + low vol regime + volume spike
            elif breakout_down and low_vol_regime and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 2.5x ATR from long extreme (trailing stop)
            # 2. Donchian breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 2.5 * atr[i]
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
            trailing_stop_short = close[i] > short_extreme + 2.5 * atr[i]
            breakout_up_exit = close[i] > donchian_upper[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_VolRegime_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0