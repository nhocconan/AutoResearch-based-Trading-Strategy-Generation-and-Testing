#!/usr/bin/env python3
"""
Experiment #5754: 1h Donchian(20) breakout + 4h/1d EMA trend filter + volume confirmation + session filter
HYPOTHESIS: Donchian breakouts on 1h aligned with 4h EMA20 and 1d EMA50 trend capture sustained moves while avoiding counter-trend whipsaws. Volume > 1.5x average confirms breakout strength. Session filter (08-20 UTC) reduces noise during low liquidity periods. Using higher timeframe EMAs as trend filter provides smoother trend detection than price action alone, reducing whipsaws in ranging markets. Discrete sizing 0.20 minimizes fees. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5754_1h_donchian20_4h_1d_ema_vol_v1"
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
    
    # === HTF: 4h data for EMA20 trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 20:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA50 trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20, 50)  # Donchian, volume avg, EMA periods
    
    for i in range(warmup, n):
        # --- Session Filter: Trade only during active hours (08-20 UTC) ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or
            np.isnan(ema_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit: price breaks below Donchian low (mean reversion) OR trend reversal
                if price <= donchian_low[i] or (price < ema_4h_aligned[i] and price < ema_1d_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit: price breaks above Donchian high (mean reversion) OR trend reversal
                if price >= donchian_high[i] or (price > ema_4h_aligned[i] and price > ema_1d_aligned[i]):
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
        
        # Trend filter: both 4h and 1d EMA must agree on direction
        long_trend = (ema_4h_aligned[i] > ema_4h_aligned[i-1]) and (ema_1d_aligned[i] > ema_1d_aligned[i-1])
        short_trend = (ema_4h_aligned[i] < ema_4h_aligned[i-1]) and (ema_1d_aligned[i] < ema_1d_aligned[i-1])
        
        # Entry conditions: breakout in direction of higher timeframe trend with volume
        long_setup = breakout_up and volume_confirmed and long_trend
        short_setup = breakout_down and volume_confirmed and short_trend
        
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