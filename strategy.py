#!/usr/bin/env python3
"""
Experiment #6334: 1h Donchian(20) breakout + 4h EMA(50) trend + session filter (08-20 UTC)
HYPOTHESIS: Tight 1h Donchian breakouts with 4h EMA(50) trend filter and session filter capture momentum trades during active hours while avoiding low-liquidity periods. Using 4h for signal direction and 1h only for entry timing reduces whipsaw and controls trade frequency. Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6334_1h_donchian20_4h_ema50_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA(50) trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        # Calculate EMA(50) on 4h close
        ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
        # Align to 1h timeframe
        ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    else:
        ema_50_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 50) + 1  # Donchian, 4h EMA + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade active hours (08-20 UTC) ---
        hour = hours[i]
        if not (8 <= hour <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit conditions:
                # 1. Price breaks below Donchian low (failed breakout)
                # 2. Price crosses below 4h EMA(50) (trend reversal)
                if price <= donchian_low[i] or price < ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit conditions:
                # 1. Price breaks above Donchian high (failed breakout)
                # 2. Price crosses above 4h EMA(50) (trend reversal)
                if price >= donchian_high[i] or price > ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        
        # Entry logic: Donchian breakout aligned with 4h EMA(50) trend
        # LONG: breakout above Donchian high + price > 4h EMA(50)
        # SHORT: breakout below Donchian low + price < 4h EMA(50)
        long_entry = breakout_up and price > ema_50_aligned[i]
        short_entry = breakout_down and price < ema_50_aligned[i]
        
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