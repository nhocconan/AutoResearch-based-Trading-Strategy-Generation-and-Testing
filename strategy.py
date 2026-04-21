#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation_v1
Hypothesis: 6h Donchian(20) breakouts filtered by weekly pivot direction (from 1w HTF) and volume confirmation (>1.5x 20-period average). Weekly pivot direction provides longer-term bias to avoid counter-trend breakouts, while volume ensures breakout conviction. Discrete position sizing (0.25) minimizes fee churn. Target 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits. Works in both bull and bear markets by aligning with weekly structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d and 1w for multi-timeframe analysis)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 5:
        return np.zeros(n)
    
    # === 1d EMA20 for intermediate trend (aligns with 6h) ===
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # === 1w pivot points for weekly directional bias ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly bias: above pivot = bullish, below = bearish
    weekly_bullish = close_1w > pivot_1w
    weekly_bearish = close_1w < pivot_1w
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # === 6h Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high/low (20-period lookback)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h volume confirmation (>1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        ema_20_1d_val = ema_20_1d_aligned[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        vol_conf = volume_confirmed[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5
        weekly_bear = weekly_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above Donchian high, weekly bullish bias, volume confirmed
            long_condition = (price > dh) and weekly_bull and vol_conf
            # Short: price breaks below Donchian low, weekly bearish bias, volume confirmed
            short_condition = (price < dl) and weekly_bear and vol_conf
            
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
                # Trend reversal exit (price below EMA20)
                elif price < ema_20_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit (optional: after 20 bars)
                elif bars_since_entry > 20:
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
                # Trend reversal exit (price above EMA20)
                elif price > ema_20_1d_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Time-based exit (optional: after 20 bars)
                elif bars_since_entry > 20:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyPivotDirection_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0