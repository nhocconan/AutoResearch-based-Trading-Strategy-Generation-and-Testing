#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA34 trend filter, volume confirmation, and chop regime filter.
- Long: price breaks above Donchian upper(20) + price > 1d EMA34 + volume > 1.5x avg + chop < 61.8
- Short: price breaks below Donchian lower(20) + price < 1d EMA34 + volume > 1.5x avg + chop < 61.8
- Exit: trailing stop (2.5x ATR from extreme) OR Donchian breakout in opposite direction
- Uses chop regime to avoid whipsaws in ranging markets
- Volume confirmation reduces false breakouts
- ATR trailing stop manages risk
- Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Chop Index(14) on 1d for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    tr_chop = np.maximum(np.abs(high[1:] - low[1:]), np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr_chop = np.concatenate([[np.nan], tr_chop])
    atr_14 = pd.Series(tr_chop).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 / (np.log10(14) * (highest_high - lowest_low)))
    chop_raw = np.where((highest_high - lowest_low) == 0, 50, chop_raw)  # avoid division by zero
    chop_1d = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 34)  # Need 20 for Donchian/volume, 14 for ATR/chop, 34 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous Donchian low
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_filter = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: Donchian breakout up + price > 1d EMA34 + volume spike + chop filter
            if breakout_up and close[i] > ema_34_aligned[i] and volume_spike and chop_filter:
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]
            # Short: Donchian breakout down + price < 1d EMA34 + volume spike + chop filter
            elif breakout_down and close[i] < ema_34_aligned[i] and volume_spike and chop_filter:
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
            breakout_down_exit = close[i] < donchian_low[i-1]
            
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
            breakout_up_exit = close[i] > donchian_high[i-1]
            
            if trailing_stop_short or breakout_up_exit:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ChopFilter_ATRStop"
timeframe = "4h"
leverage = 1.0