#!/usr/bin/env python3
"""
Experiment #6434: 1h Donchian(20) breakout with 4h/1d trend and volume confirmation
HYPOTHESIS: 1h Donchian breakouts with volume confirmation (>2.0x avg) and multi-timeframe trend filter (4h/1d EMA50) capture momentum while minimizing false signals. 
Uses 4h/1d for signal direction, 1h only for entry timing. Session filter (08-20 UTC) reduces noise. 
Discrete sizing at 0.20 to control fee drag. Target: 60-150 total trades over 4 years.
Works in bull via upward breakouts with volume and uptrend, in bear via downward breakdowns with volume and downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6434_1h_donchian20_4h_1d_ema_vol_v1"
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
    
    # === HTF: 4h data for EMA21 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        # Calculate 4h EMA21
        ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)  # auto shift(1) for completed bars only
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        # Calculate 1d EMA50
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)  # auto shift(1) for completed bars only
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 1h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21, 50) + 1  # Donchian, volume avg, ATR, 4h EMA, 1d EMA lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Trade only during active hours (08:00-20:00 UTC) ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
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
                if price <= stop_price or price <= donchian_low[i]:
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
        volume_confirmed = volume_ratio[i] > 2.0  # Volume filter
        
        # Entry logic based on 4h/1d EMA trend:
        # Long: breakout above Donchian high with volume AND price > 4h EMA21 AND price > 1d EMA50 (strong uptrend)
        # Short: breakdown below Donchian low with volume AND price < 4h EMA21 AND price < 1d EMA50 (strong downtrend)
        long_entry = breakout_up and volume_confirmed and (price > ema_4h_aligned[i]) and (price > ema_1d_aligned[i])
        short_entry = breakout_down and volume_confirmed and (price < ema_4h_aligned[i]) and (price < ema_1d_aligned[i])
        
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