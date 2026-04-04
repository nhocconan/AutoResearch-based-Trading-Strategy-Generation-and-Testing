#!/usr/bin/env python3
"""
Experiment #6157: 4h Donchian(20) breakout + 1d volume confirmation + ATR trailing stop
HYPOTHESIS: 4h Donchian breakouts with volume confirmation capture institutional participation in both bull and bear markets. Using 1d HTF for volume filter ensures alignment with higher timeframe participation. ATR-based trailing stop manages risk during volatile periods. Discrete sizing (0.25) minimizes fee churn. Target: 100-180 trades over 4 years.
Timeframe: 4h. HTF: 1d for volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6157_4h_donchian20_1d_vol_confirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d volume for confirmation (use prior day's average volume) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Daily volume average (use same day's volume for 4h alignment)
        daily_volume = df_1d['volume'].values
        # Shift by 1 to avoid look-ahead: use previous day's volume for current 4h bars
        daily_volume_shifted = np.roll(daily_volume, 1)
        daily_volume_shifted[0] = np.nan  # First value undefined
        # Align to 4h timeframe
        volume_1d_avg_aligned = align_htf_to_ltf(prices, df_1d, daily_volume_shifted)
    else:
        volume_1d_avg_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 14) + 1  # Donchian, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21:00-23:59 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume[i]) or np.isnan(atr[i]) or
            np.isnan(volume_1d_avg_aligned[i])):
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
        # Volume confirmation: current 4h volume > 1.5x previous day's average volume
        volume_confirmed = volume[i] > 1.5 * volume_1d_avg_aligned[i]
        
        # Entry conditions:
        # Long: breakout up with volume confirmation
        # Short: breakout down with volume confirmation
        long_entry = breakout_up and volume_confirmed
        short_entry = breakout_down and volume_confirmed
        
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