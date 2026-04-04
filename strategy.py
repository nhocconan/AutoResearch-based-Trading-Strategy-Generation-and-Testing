#!/usr/bin/env python3
"""
Experiment #2707: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot bias (price above/below weekly pivot)
and volume confirmation capture institutional participation with lower frequency than 4h strategies.
Uses 1d for weekly pivot calculation (resampled to weekly using actual Binance 1d data), 6h only for entry.
Target: 75-150 total trades over 4 years. Works in bull/bear via pivot bias filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2707_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points from 1d data (using actual Binance 1d candles)
    # We need to group into weeks: Monday 00:00 UTC to Sunday 24:00 UTC
    # Since df_1d is already daily, we can calculate weekly OHLC
    df_1d_indexed = df_1d.copy()
    df_1d_indexed.index = pd.date_range(
        start=prices['open_time'].iloc[0], 
        periods=len(df_1d), 
        freq='1d'
    )
    
    # Resample to weekly (W-MON: weekly starting Monday)
    weekly = df_1d_indexed.resample('W-MON').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    if len(weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot: P = (H + L + C) / 3
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 1d (each day gets the pivot from the week it belongs to)
    # We need to map each 1d bar to its weekly pivot
    weekly_pivot_series = pd.Series(weekly_pivot, index=weekly.index)
    # Forward fill to get pivot for each day in the week
    daily_index = df_1d_indexed.index
    weekly_pivot_daily = weekly_pivot_series.reindex(daily_index, method='ffill').values
    
    # Align weekly pivot from 1d to 6h
    weekly_pivot_6h = align_htf_to_ltf(prices, df_1d, weekly_pivot_daily)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
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
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot_6h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        weekly_pivot = weekly_pivot_6h[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require price to be on correct side of weekly pivot for bias filter
        # Long bias: price above weekly pivot
        # Short bias: price below weekly pivot
        price_above_pivot = price > weekly_pivot
        price_below_pivot = price < weekly_pivot
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND price above weekly pivot
            if price_above_pivot and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND price below weekly pivot
            elif price_below_pivot and price < lowest_20[i]:
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