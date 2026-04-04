#!/usr/bin/env python3
"""
Experiment #6359: 6h Donchian(20) breakout + 12h Supertrend + volume confirmation
HYPOTHESIS: 6h Donchian breakouts with volume confirmation (>1.8x average) and 12h Supertrend filter capture institutional momentum while avoiding whipsaws. 
Supertrend on 12h provides robust trend direction that works in both bull and bear markets by adapting to volatility via ATR. 
Volume confirmation ensures breakouts have participation. Discrete sizing (0.25) minimizes fee churn. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6359_6h_donchian20_12h_supertrend_vol_v1"
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
    
    # === HTF: 12h data for Supertrend ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 10:
        # Supertrend calculation
        atr_period = 10
        multiplier = 3.0
        
        # True Range
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
        tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
        
        # Basic Upper and Lower Bands
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        basic_ub = hl2 + (multiplier * atr)
        basic_lb = hl2 - (multiplier * atr)
        
        # Final Upper and Lower Bands
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        for i in range(1, len(basic_ub)):
            if basic_ub[i] < final_ub[i-1] or df_12h['close'].iloc[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or df_12h['close'].iloc[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        # Supertrend
        supertrend = np.full(len(df_12h), np.nan)
        for i in range(atr_period, len(df_12h)):
            if i == atr_period:
                supertrend[i] = final_ub.iloc[i]
            else:
                if supertrend[i-1] == final_ub.iloc[i-1] and df_12h['close'].iloc[i] <= final_ub.iloc[i]:
                    supertrend[i] = final_ub.iloc[i]
                elif supertrend[i-1] == final_ub.iloc[i-1] and df_12h['close'].iloc[i] > final_ub.iloc[i]:
                    supertrend[i] = final_lb.iloc[i]
                elif supertrend[i-1] == final_lb.iloc[i-1] and df_12h['close'].iloc[i] >= final_lb.iloc[i]:
                    supertrend[i] = final_lb.iloc[i]
                elif supertrend[i-1] == final_lb.iloc[i-1] and df_12h['close'].iloc[i] < final_lb.iloc[i]:
                    supertrend[i] = final_ub.iloc[i]
        
        # Align to 6h timeframe
        supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend.values)
    else:
        supertrend_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 14, 10) + 1  # Donchian, volume avg, ATR, Supertrend warmup + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(supertrend_aligned[i])):
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
                # 3. Supertrend turns bearish
                if price <= stop_price or price <= donchian_low[i] or price < supertrend_aligned[i]:
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
                # 3. Supertrend turns bullish
                if price >= stop_price or price >= donchian_high[i] or price > supertrend_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8  # Volume filter
        
        long_entry = False
        short_entry = False
        
        # Only trade in direction of 12h Supertrend
        if breakout_up and volume_confirmed and price > supertrend_aligned[i]:
            long_entry = True
        if breakout_down and volume_confirmed and price < supertrend_aligned[i]:
            short_entry = True
        
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