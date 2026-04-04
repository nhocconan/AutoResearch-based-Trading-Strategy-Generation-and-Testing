#!/usr/bin/env python3
"""
Experiment #6291: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation, filtered by 1d weekly pivot
direction (price above/below weekly pivot), capture institutional momentum while avoiding
counter-trend whipsaws. Weekly pivot provides structural support/resistance from higher timeframe.
Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
Uses discrete sizing (0.25) and ATR trailing stop for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6291_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    if len(df_1d) >= 5:  # Need at least a week for weekly pivot
        # Calculate weekly pivot from 1d OHLC (using prior week's data)
        # Resample 1d to weekly using actual weekly boundaries (not rolling)
        df_1d_indexed = pd.DataFrame({
            'open': df_1d['open'],
            'high': df_1d['high'],
            'low': df_1d['low'],
            'close': df_1d['close']
        }, index=pd.to_datetime(df_1d['open_time']))
        
        # Weekly resample: get prior completed week's OHLC
        weekly_ohlc = df_1d_indexed.resample('W-Wed').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).shift(1)  # Use prior week only (no look-ahead)
        
        if len(weekly_ohlc) > 0:
            # Weekly pivot: P = (H + L + C) / 3
            weekly_high = weekly_ohlc['high'].values
            weekly_low = weekly_ohlc['low'].values
            weekly_close = weekly_ohlc['close'].values
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
            
            # Align weekly pivot to 6h timeframe
            weekly_pivot_aligned = align_htf_to_ltf(prices, weekly_ohlc, weekly_pivot.values)
        else:
            weekly_pivot_aligned = np.full(n, np.nan)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below weekly pivot (trend change)
                if price <= stop_price or price <= donchian_low[i] or price < weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above weekly pivot (trend change)
                if price >= stop_price or price >= donchian_high[i] or price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Strong volume filter
        
        # Entry logic: Donchian breakout with volume AND aligned with weekly pivot
        # LONG: breakout above Donchian high + volume + price > weekly pivot
        # SHORT: breakout below Donchian low + volume + price < weekly pivot
        long_entry = breakout_up and volume_confirmed and price > weekly_pivot_aligned[i]
        short_entry = breakout_down and volume_confirmed and price < weekly_pivot_aligned[i]
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals