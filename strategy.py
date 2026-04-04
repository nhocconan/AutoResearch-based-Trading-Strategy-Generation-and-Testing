#!/usr/bin/env python3
"""
Experiment #6349: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
HYPOTHESIS: Tight 4h Donchian breakouts with 1d volume confirmation (>2.0x average) and choppiness regime (CHOP > 61.8 = range) capture strong momentum moves while avoiding whipsaw in both bull and bear markets. Uses discrete sizing (0.30) to minimize fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6349_4h_donchian20_1d_vol_chop_v1"
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
    
    # === HTF: 1d data for volume and chop regime ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # 1d ATR for chop calculation
        tr1 = df_1d['high'] - df_1d['low']
        tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
        tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
        tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
        tr_1d[0] = tr1.iloc[0] if hasattr(tr1, 'iloc') else tr1[0]
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        
        # 1d true range sum for chop denominator
        tr_sum = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
        
        # 1d high-low range for chop numerator
        hh_1d = df_1d['high'].rolling(window=14, min_periods=14).max().values
        ll_1d = df_1d['low'].rolling(window=14, min_periods=14).min().values
        range_1d = hh_1d - ll_1d
        
        # Choppiness Index: CHOP = 100 * log10(tr_sum / range_1d) / log10(14)
        # CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
        chop_raw = 100 * np.log10(tr_sum / np.where(range_1d > 0, range_1d, 1e-10)) / np.log10(14)
        chop_raw = np.nan_to_num(chop_raw, nan=50.0)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
        
        # 1d volume average for confirmation
        avg_volume_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
        volume_ratio_1d = df_1d['volume'].values / np.where(avg_volume_1d > 0, avg_volume_1d, 1)
        volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    else:
        chop_aligned = np.full(n, 50.0)
        volume_ratio_1d_aligned = np.full(n, 1.0)
    
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
    SIZE = 0.30  # 30% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14) + 1  # Donchian, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_ratio_1d_aligned[i])):
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
                # 3. Chop regime shifts to trending (CHOP < 38.2) - take profit in trend
                if price <= stop_price or price <= donchian_low[i] or chop_aligned[i] < 38.2:
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
                # 3. Chop regime shifts to trending (CHOP < 38.2) - take profit in trend
                if price >= stop_price or price >= donchian_high[i] or chop_aligned[i] < 38.2:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio_1d_aligned[i] > 2.0  # Volume spike filter
        chop_range = chop_aligned[i] > 61.8  # Range regime for mean reversion bias
        
        # Entry logic: Donchian breakout with volume confirmation in ranging markets
        # LONG: breakout above Donchian high + volume spike + chop > 61.8 (range)
        # SHORT: breakout below Donchian low + volume spike + chop > 61.8 (range)
        long_entry = breakout_up and volume_confirmed and chop_range
        short_entry = breakout_down and volume_confirmed and chop_range
        
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