#!/usr/bin/env python3
"""
Experiment #5307: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.5x average and aligned with weekly pivot bias (price above/below weekly pivot) 
captures strong momentum moves with institutional participation. Weekly pivot provides 
structure from higher timeframe (1w) that works in both bull and bear markets by identifying 
key support/resistance levels where breakouts have higher follow-through. Volume confirmation 
ensures breakouts have genuine participation rather than low-liquidity spikes. 
Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5307_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate weekly pivot from daily OHLC (using prior week's data)
        # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        df_1d = df_1d.copy()
        df_1d['week'] = pd.DatetimeIndex(df_1d.index).isocalendar().week
        df_1d['year'] = pd.DatetimeIndex(df_1d.index).isocalendar().year
        
        # Group by year-week to get weekly OHLC
        weekly = df_1d.groupby(['year', 'week']).agg({
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        
        if len(weekly) >= 2:
            # Shift by 1 to use prior week's data (no look-ahead)
            weekly['pivot'] = (weekly['high'].shift(1) + weekly['low'].shift(1) + weekly['close'].shift(1)) / 3
            weekly['prior_week_high'] = weekly['high'].shift(1)
            weekly['prior_week_low'] = weekly['low'].shift(1)
            
            # Map back to daily data
            weekly_map = weekly.set_index(['year', 'week'])
            df_1d['year_week'] = list(zip(
                pd.DatetimeIndex(df_1d.index).isocalendar().year,
                pd.DatetimeIndex(df_1d.index).isocalendar().week
            ))
            df_1d['pivot'] = df_1d['year_week'].map(weekly_map['pivot'])
            df_1d['prior_week_high'] = df_1d['year_week'].map(weekly_map['prior_week_high'])
            df_1d['prior_week_low'] = df_1d['year_week'].map(weekly_map['prior_week_low'])
            
            # Align to 6h timeframe
            pivot_vals = df_1d['pivot'].values
            prior_week_high_vals = df_1d['prior_week_high'].values
            prior_week_low_vals = df_1d['prior_week_low'].values
            
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
            prior_week_high_aligned = align_htf_to_ltf(prices, df_1d, prior_week_high_vals)
            prior_week_low_aligned = align_htf_to_ltf(prices, df_1d, prior_week_low_vals)
        else:
            pivot_aligned = np.full(n, np.nan)
            prior_week_high_aligned = np.full(n, np.nan)
            prior_week_low_aligned = np.full(n, np.nan)
    else:
        pivot_aligned = np.full(n, np.nan)
        prior_week_high_aligned = np.full(n, np.nan)
        prior_week_low_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    # Upper band: 20-period high
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    # Average volume over 20 periods
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)  # Avoid division by zero
    
    # === 6h Indicators: ATR(14) for stoploss ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Trade during major sessions only ---
        hour = hours[i]
        # Trade during major sessions: 00-06 UTC (Asia), 07-12 UTC (Europe), 13-20 UTC (US)
        # Avoid 21-23 UTC (low liquidity between sessions)
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or failed breakout ---
        if in_position:
            # Update highest/lowest since entry for trailing stop logic
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Price falls below weekly pivot (bias change)
                if price <= stop_price or price <= donchian_low[i] or price < pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Price rises above weekly pivot (bias change)
                if price >= stop_price or price >= donchian_high[i] or price > pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Bias from weekly pivot: price above pivot = bullish bias, below = bearish bias
        bullish_bias = price > pivot_aligned[i]
        bearish_bias = price < pivot_aligned[i]
        
        # Entry conditions: Donchian breakout + volume confirmation + pivot bias alignment
        if breakout_up and volume_confirmed and bullish_bias:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and bearish_bias:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals