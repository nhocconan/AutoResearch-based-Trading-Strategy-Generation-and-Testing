#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ADX(14) trend filter and volume confirmation
- Uses Donchian channel (20-period high/low) from 6h timeframe for breakout signals
- 1d ADX(14) > 25 defines strong trend (only trade in direction of trend)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend direction
- ADX filter prevents whipsaws in ranging markets, volume confirms momentum
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
    
    # Calculate 6h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 1d ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period_adx = 14
    if len(tr) >= period_adx:
        atr = WilderSmoothing(tr, period_adx)
        plus_di = 100 * WilderSmoothing(plus_dm, period_adx) / atr
        minus_di = 100 * WilderSmoothing(minus_dm, period_adx) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = WilderSmoothing(dx, period_adx)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    # Add padding for alignment (ADX starts at index 2*period_adx-1)
    adx_padded = np.full(len(close_1d), np.nan)
    if len(adx) > 0:
        start_idx = 2 * period_adx - 1
        end_idx = start_idx + len(adx)
        if end_idx <= len(adx_padded):
            adx_padded[start_idx:end_idx] = adx
    
    # Align indicators to 6h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_padded, additional_delay_bars=0)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 2*period_adx-1+period_adx, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_donchian_high = close[i] > highest_high_aligned[i]
        price_below_donchian_low = close[i] < lowest_low_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, strong trend, volume spike
            long_signal = (price_above_donchian_high and 
                          strong_trend and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below Donchian low, strong trend, volume spike
            short_signal = (price_below_donchian_low and 
                           strong_trend and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or trend weakens (ADX < 20)
                if (price_below_donchian_low or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend weakens
                if (price_above_donchian_high or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_Breakout_1dADX14_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0