#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (bullish if price > weekly pivot, bearish if <) and volume confirmation (1.5x 20-period average) capture sustained moves in both bull and bear markets. Weekly pivot provides structural bias, Donchian breaks momentum, volume filters noise. Discrete sizing (0.25) and ATR stop (2.0x) minimize fee drag. Target: 50-150 total trades over 4 years for BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for daily pivot, 1w for weekly trend)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === Weekly pivot direction (bullish if close > weekly pivot) ===
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower = 20-period rolling max/min
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        vol_conf = volume_confirmed[i]
        
        # Weekly pivot direction bias
        is_bullish_bias = price > weekly_pivot_val
        is_bearish_bias = price < weekly_pivot_val
        
        if position == 0:
            # Long: price breaks above Donchian high with bullish bias and volume
            long_condition = (price > donch_high_val) and is_bullish_bias and vol_conf
            # Short: price breaks below Donchian low with bearish bias and volume
            short_condition = (price < donch_low_val) and is_bearish_bias and vol_conf
            
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
                # Exit if price breaks below Donchian low (failed breakout)
                elif price < donch_low_val:
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
                # Exit if price breaks above Donchian high (failed breakdown)
                elif price > donch_high_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0