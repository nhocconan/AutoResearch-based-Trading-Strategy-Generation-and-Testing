#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 20-period high, weekly pivot is bullish (price > weekly pivot), and volume > 2.0x 20-period MA.
Short when price breaks below 20-period low, weekly pivot is bearish (price < weekly pivot), and volume > 2.0x 20-period MA.
Uses ATR-based stop (2.0x) and minimum holding period of 3 bars to reduce churn.
Designed for low trade frequency (~15-25/year) to work in both bull and bear markets via weekly structural bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for weekly pivot via daily resample approximation, 1w for true weekly)
    # We'll use 1d to approximate weekly pivot by taking last 5 daily bars (simplified)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === Weekly pivot approximation from daily data (using last 5 daily bars) ===
    # True weekly pivot requires weekly data, but we approximate with recent daily range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use last 5 daily bars to approximate weekly range (reduces lag)
    lookback = 5
    if len(high_1d) < lookback:
        weekly_high = np.full_like(close_1d, np.nan)
        weekly_low = np.full_like(close_1d, np.nan)
        weekly_close = np.full_like(close_1d, np.nan)
    else:
        # Rolling max/min/close over 5 days
        weekly_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
        weekly_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
        weekly_close = pd.Series(close_1d).rolling(window=lookback, min_periods=lookback).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 6h Donchian channels (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        vol_avg = vol_ma[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirm = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price breaks above Donchian high, above weekly pivot, volume confirm
            long_condition = (price > donchian_high_val) and (price > weekly_pivot_val) and volume_confirm
            # Short: price breaks below Donchian low, below weekly pivot, volume confirm
            short_condition = (price < donchian_low_val) and (price < weekly_pivot_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Pivot reversal exit (price below weekly pivot)
                elif price < weekly_pivot_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Pivot reversal exit (price above weekly pivot)
                elif price > weekly_pivot_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike"
timeframe = "6h"
leverage = 1.0