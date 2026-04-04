#!/usr/bin/env python3
"""
Experiment #5899: 6h Donchian(20) breakout + 12h Supertrend regime + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h Supertrend regime capture high-probability 
continuation moves in both bull and bear markets. Volume confirmation filters weak breakouts. 
Supertrend (12h) provides adaptive trend filtering that works across regimes. Target: 75-150 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5899_6h_donchian20_12h_supertrend_vol_v1"
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
    
    # === HTF: 12h data for Supertrend regime ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Calculate Supertrend on 12h data
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
        tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr_12h = tr.rolling(window=10, min_periods=10).mean()
        
        upper_band = hl2 + (3.0 * atr_12h)
        lower_band = hl2 - (3.0 * atr_12h)
        
        # Initialize Supertrend
        supertrend = pd.Series(index=df_12h.index, dtype=float)
        direction = pd.Series(index=df_12h.index, dtype=float)  # 1 for uptrend, -1 for downtrend
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df_12h)):
            close_price = df_12h['close'].iloc[i]
            
            if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                supertrend.iloc[i] = lower_band.iloc[i] if close_price <= upper_band.iloc[i-1] else upper_band.iloc[i]
                direction.iloc[i] = -1 if supertrend.iloc[i] == lower_band.iloc[i] else 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i] if close_price >= lower_band.iloc[i-1] else lower_band.iloc[i]
                direction.iloc[i] = 1 if supertrend.iloc[i] == upper_band.iloc[i] else -1
        
        # Align Supertrend direction to LTF (6h) with shift(1) for completed bars only
        supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction.values)
    else:
        supertrend_dir_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 10, 14)  # Donchian, volume avg, Supertrend, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(supertrend_dir_aligned[i])):
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
        
        # Supertrend regime logic:
        # Supertrend direction = 1 (uptrend) = long bias
        # Supertrend direction = -1 (downtrend) = short bias
        long_bias = supertrend_dir_aligned[i] == 1
        short_bias = supertrend_dir_aligned[i] == -1
        
        # Entry conditions: breakout in direction of Supertrend regime
        long_setup = breakout_up and volume_confirmed and long_bias
        short_setup = breakout_down and volume_confirmed and short_bias
        
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