#!/usr/bin/env python3
"""
Experiment #2883: 4h Donchian Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends with controlled frequency.
12-period HMA on 12h timeframe provides trend filter: only long when HMA rising, short when falling.
Volume spike (>2.0x 20-period average) confirms breakout strength. ATR-based trailing stop (2.5x)
manages risk. Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
Works in both bull/bear via trend filter and symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2883_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(12) on 12h: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw = 2 * wma_half - wma_full
        hma = pd.Series(raw).rolling(window=sqrt, min_periods=sqrt).mean().values
        return hma
    
    hma_12h = calculate_hma(close_12h, 12)
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    # HMA trend: 1 if rising, -1 if falling, 0 if flat/undefined
    hma_trend = np.zeros(n)
    hma_trend[1:] = np.where(hma_12h_aligned[1:] > hma_12h_aligned[:-1], 1,
                              np.where(hma_12h_aligned[1:] < hma_12h_aligned[:-1], -1, 0))
    
    # === 4h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_trend[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Approximate ATR from price range (conservative)
                atr_estimate = (high[i] - low[i]) * 0.5
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Donchian trend reverses (price re-enters channel)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                atr_estimate = (high[i] - low[i]) * 0.5
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if Donchian trend reverses
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) AND trend alignment
        volume_spike = vol_ratio[i] > 2.0
        trend_aligned = hma_trend[i] != 0
        
        if volume_spike and trend_aligned:
            # Long entry: price breaks above Donchian high with rising HMA
            if price > highest_high[i] and hma_trend[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with falling HMA
            elif price < lowest_low[i] and hma_trend[i] < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals