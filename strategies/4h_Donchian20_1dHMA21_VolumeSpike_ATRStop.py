#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d HMA(21) trend filter, volume confirmation (1.8x 20-period average), and ATR(14) trailing stop (2.5x).
- Long: price breaks above Donchian upper (20-period high) + price > 1d HMA21 + volume > 1.8x 20-period avg volume
- Short: price breaks below Donchian lower (20-period low) + price < 1d HMA21 + volume > 1.8x 20-period avg volume
- Exit: trailing stop (2.5x ATR from extreme) OR Donchian breakout in opposite direction
- Uses 1d HMA21 as trend filter to avoid counter-trend trades and adapt to regime
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: trend filter adapts to regime
- Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
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
    
    # Calculate Donchian channels (20-period) using previous bar to avoid look-ahead
    # Upper = max(high over last 20 periods)
    # Lower = min(low over last 20 periods)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: > 1.8x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d HMA21 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def calculate_wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def calculate_hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = calculate_wma(values, half_window)
        wma_full = calculate_wma(values, window)
        wma_2x_half = 2 * wma_half
        # Pad arrays to same length
        diff = wma_2x_half[-len(wma_full):] - wma_full
        hma = calculate_wma(diff, sqrt_window)
        # Pad beginning with NaN
        hma_padded = np.full_like(values, np.nan)
        hma_padded[-len(hma):] = hma
        return hma_padded
    
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 21)  # Need 20 for Donchian, 14 for ATR, 21 for HMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(hma_21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_upper[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 1.8x average)
        volume_spike = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d HMA21 + volume spike
            if breakout_up and close[i] > hma_21_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + price < 1d HMA21 + volume spike
            elif breakout_down and close[i] < hma_21_aligned[i] and volume_spike:
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

name = "4h_Donchian20_1dHMA21_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0