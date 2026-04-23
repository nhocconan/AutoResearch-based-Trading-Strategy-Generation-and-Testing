#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop
- Long when: price breaks above 20-period high + price > 1d EMA50 + volume > 1.5x 20-period average
- Short when: price breaks below 20-period low + price < 1d EMA50 + volume > 1.5x 20-period average
- Exit when: price reverses 2.5x ATR from extreme (trailing stop) OR opposite Donchian breakout
- Uses 1d EMA50 as trend filter to align with higher timeframe momentum
- Volume confirmation (1.5x average) reduces false breakouts while maintaining sufficient trade frequency
- ATR trailing stop manages risk without look-ahead
- Designed for both bull and bear markets: trend filter adapts to regime
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
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
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d EMA50 ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for Donchian/volume, 14 for ATR, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's bands)
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous period's upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous period's lower band
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA50 + volume confirmation
            if breakout_up and close[i] > ema_50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + price < 1d EMA50 + volume confirmation
            elif breakout_down and close[i] < ema_50_aligned[i] and volume_confirm:
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
            breakout_down_exit = close[i] < donchian_lower[i-1]
            
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
            breakout_up_exit = close[i] > donchian_upper[i-1]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_VolumeConfirm_ATRStop"
timeframe = "12h"
leverage = 1.0