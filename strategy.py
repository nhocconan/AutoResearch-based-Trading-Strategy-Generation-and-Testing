#!/usr/bin/env python3
"""
Experiment #4507: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with 1d weekly pivot direction (price above weekly pivot = long bias, below = short bias) and confirmed by volume (>2.0x average) capture medium-term momentum with reduced noise. Weekly pivot provides structural support/resistance from higher timeframe to avoid counter-trend trades, while volume confirmation ensures breakout conviction. Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year) with position size 0.25. Works in both bull and bear markets by only trading in direction of weekly pivot bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4507_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot (using prior week's H/L/C)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:  # Need at least 5 days for prior week
        # Calculate weekly pivot from prior week's data (shifted by 1 to avoid look-ahead)
        # We'll use the prior week's high, low, close to compute pivot for current week
        high_1d = pd.Series(df_1d['high'].values)
        low_1d = pd.Series(df_1d['low'].values)
        close_1d = pd.Series(df_1d['close'].values)
        
        # Weekly aggregates: prior week's H/L/C (shifted by 5 days to get prior week, then another 1 to avoid look-ahead)
        # Actually simpler: use rolling window of 5 days, shift by 1 to get completed week
        week_high = high_1d.rolling(window=5, min_periods=5).max().shift(1)  # Prior week's high
        week_low = low_1d.rolling(window=5, min_periods=5).min().shift(1)    # Prior week's low
        week_close = close_1d.rolling(window=5, min_periods=5).last().shift(1) # Prior week's close
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        weekly_pivot_vals = weekly_pivot.values
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_vals)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 5)  # Donchian, vol MA, ATR, 1d weekly data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Weekly pivot bias: price above pivot = long bias, below = short bias
        long_bias = price > weekly_pivot_aligned[i]
        short_bias = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + long bias + volume
        long_entry = breakout_up and long_bias and volume_confirm
        
        # Short conditions: downward breakout + short bias + volume
        short_entry = breakout_down and short_bias and volume_confirm
        
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