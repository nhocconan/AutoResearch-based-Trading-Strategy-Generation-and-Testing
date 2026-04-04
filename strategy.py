#!/usr/bin/env python3
"""
Experiment #6079: 6h Donchian(20) breakout + 12h ADX trend + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h ADX>25 trend capture strong momentum moves. 
ADX filters out choppy/range-bound markets where breakouts fail. Volume >1.5x average confirms 
institutional participation. Works in bull markets (breakouts with rising ADX) and bear markets 
(breakdowns with rising ADX). Target: 75-200 trades over 4 years (19-50/year). Discrete sizing 
(0.25) minimizes fee drag. Uses proper MTF via mtf_data helper.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6079_6h_donchian20_12h_adx_vol_v1"
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
    
    # === HTF: 12h data for ADX(14) trend strength ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 14:
        # Calculate ADX components
        plus_dm = np.diff(df_12h['high'].values, prepend=df_12h['high'].values[0])
        minus_dm = np.diff(df_12h['low'].values, prepend=df_12h['low'].values[0]) * -1
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
        
        tr1 = df_12h['high'].values - df_12h['low'].values
        tr2 = np.abs(np.diff(df_12h['high'].values, prepend=df_12h['high'].values[0]))
        tr3 = np.abs(np.diff(df_12h['low'].values, prepend=df_12h['low'].values[0]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_12h > 0, atr_12h, 1)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_12h > 0, atr_12h, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
        adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        
        adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 14) + 1  # Donchian, volume avg, ADX, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.0 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.0 * atr[i]
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
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter for stronger signals
        strong_trend = adx_aligned[i] > 25  # ADX > 25 indicates strong trend
        
        # Entry conditions:
        # Long: breakout up with volume AND strong trend
        # Short: breakout down with volume AND strong trend
        long_entry = breakout_up and volume_confirmed and strong_trend
        short_entry = breakout_down and volume_confirmed and strong_trend
        
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