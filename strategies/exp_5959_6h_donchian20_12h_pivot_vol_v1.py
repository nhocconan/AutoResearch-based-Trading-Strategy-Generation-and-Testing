#!/usr/bin/env python3
"""
Experiment #5959: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with 12h pivot (price vs 12h pivot) capture sustained moves with lower noise than daily.
12h pivot provides structural support/resistance from higher timeframe (more stable than 1d on 6h chart). Volume >1.5x average confirms breakout strength.
ATR trailing stop manages risk. Target: 75-150 trades over 4 years (19-37/year) to minimize fee drift.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5959_6h_donchian20_12h_pivot_vol_v1"
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
    
    # === HTF: 12h data for pivot ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 1:
        # Calculate pivot from prior 12h bar's OHLC
        high_12h = pd.Series(df_12h['high'].values)
        low_12h = pd.Series(df_12h['low'].values)
        close_12h = pd.Series(df_12h['close'].values)
        
        # Prior 12h OHLC
        prior_high = high_12h.shift(1)
        prior_low = low_12h.shift(1)
        prior_close = close_12h.shift(1)
        
        # Pivot point: (H + L + C) / 3
        pivot_12h = (prior_high + prior_low + prior_close) / 3.0
        pivot_12h_values = pivot_12h.values
        pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h_values)
    else:
        pivot_12h_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 1) + 1  # Donchian, volume avg, ATR, pivot shift + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Pivot filter: price above/below 12h pivot
        above_pivot = price > pivot_12h_aligned[i]
        below_pivot = price < pivot_12h_aligned[i]
        
        # Entry conditions: 
        # Long: breakout up with volume AND above 12h pivot
        # Short: breakout down with volume AND below 12h pivot
        long_setup = breakout_up and volume_confirmed and above_pivot
        short_setup = breakout_down and volume_confirmed and below_pivot
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals