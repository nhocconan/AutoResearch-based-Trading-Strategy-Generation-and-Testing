#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike_ATRStop_v1
Hypothesis: 6h Donchian(20) breakouts filtered by weekly pivot direction (from 1w HTF) and volume confirmation.
In bull markets (price > weekly pivot), take long breakouts; in bear markets (price < weekly pivot), take short breakouts.
Volume filter (>1.5x 20-period average) reduces false breakouts. ATR-based stoploss (1.5x ATR) manages risk.
Designed for 12-37 trades/year per symbol (~50-150 total over 4 years) to minimize fee drag.
Uses weekly pivot as regime filter to adapt to bull/bear conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === 6h OHLC for Donchian calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly pivot for regime filter ===
    # Calculate weekly pivot from prior week's H/L/C
    df_1w_copy = df_1w.copy()
    df_1w_copy['week_start'] = df_1w_copy.index.to_series().dt.to_period('W').dt.start_time
    weekly_high = df_1w_copy.groupby('week_start')['high'].shift(1).values
    weekly_low = df_1w_copy.groupby('week_start')['low'].shift(1).values
    weekly_close = df_1w_copy.groupby('week_start')['close'].shift(1).values
    
    # Weekly pivot = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(weekly_pivot_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.5x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Determine regime: bullish if price > weekly pivot, bearish if price < weekly pivot
            bullish_regime = price > weekly_pivot_aligned[i]
            bearish_regime = price < weekly_pivot_aligned[i]
            
            # Long conditions: price > Donchian high (breakout), bullish regime, volume filter
            long_breakout = price > donchian_high[i]
            
            # Short conditions: price < Donchian low (breakdown), bearish regime, volume filter
            short_breakout = price < donchian_low[i]
            
            # Entry logic - ONLY enter on volume filter + regime alignment
            if long_breakout and bullish_regime and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and bearish_regime and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below Donchian low (breakdown)
            elif price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above Donchian high (breakout)
            elif price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivotDirection_VolumeSpike_ATRStop_v1"
timeframe = "6h"
leverage = 1.0