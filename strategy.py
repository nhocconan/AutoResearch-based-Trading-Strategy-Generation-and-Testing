#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (2.0x 20-period average), and ATR(14) trailing stop (3.0x).
- Long: price breaks above Donchian upper (20-period high) + price > 1d EMA34 + volume > 2.0x 20-period average volume
- Short: price breaks below Donchian lower (20-period low) + price < 1d EMA34 + volume > 2.0x 20-period average volume
- Exit: trailing stop (3.0x ATR from extreme) OR Donchian breakout in opposite direction
- Uses 1d EMA34 as trend filter to avoid counter-trend trades
- Volume spike reduces false breakouts
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
    
    # Calculate Donchian channels (20-period)
    # Upper = highest high of past 20 periods (excluding current)
    # Lower = lowest low of past 20 periods (excluding current)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: > 2.0x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA34 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for Donchian, 14 for ATR, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous period's levels)
        breakout_up = close[i] > donchian_upper[i]  # Break above Donchian upper
        breakout_down = close[i] < donchian_lower[i]  # Break below Donchian lower
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA34 + volume spike
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + price < 1d EMA34 + volume spike
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, high[i])
            
            # Exit conditions:
            # 1. Price reverses 3.0x ATR from long extreme (trailing stop)
            # 2. Donchian breakout down (opposite signal)
            trailing_stop_long = close[i] < long_extreme - 3.0 * atr[i]
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
            # 1. Price reverses 3.0x ATR from short extreme (trailing stop)
            # 2. Donchian breakout up (opposite signal)
            trailing_stop_short = close[i] > short_extreme + 3.0 * atr[i]
            breakout_up_exit = close[i] > donchian_upper[i]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0