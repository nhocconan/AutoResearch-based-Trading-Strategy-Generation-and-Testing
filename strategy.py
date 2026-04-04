#!/usr/bin/env python3
"""
Experiment #5781: 4h Donchian(20) breakout + 1d Camarilla pivot continuation + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d Camarilla R4/S4 levels (strong breakout zones) capture explosive moves with volume confirmation. Uses 1d timeframe for structure to reduce whipsaws, targeting 75-200 trades over 4 years. Works in bull/bear markets by requiring breakout alignment with pivot-derived support/resistance. Discrete sizing 0.25 minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5781_4h_donchian20_1d_camarilla_vol_v1"
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
    
    # === HTF: 1d data for Camarilla pivot levels ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 2:
        # Calculate Camarilla pivot levels from previous 1d bar
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        pivot = (high_1d + low_1d + close_1d) / 3.0
        range_1d = high_1d - low_1d
        # Camarilla levels: R4 = close + range * 1.1/2, S4 = close - range * 1.1/2
        camarilla_r4 = close_1d + range_1d * 1.1 / 2.0
        camarilla_s4 = close_1d - range_1d * 1.1 / 2.0
    else:
        camarilla_r4 = np.full(len(df_1d), np.nan)
        camarilla_s4 = np.full(len(df_1d), np.nan)
    
    # Align 1d Camarilla levels to 4h timeframe (shifted by 1 for completed 1d bars only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
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
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, need at least 2 for 1d pivot
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Camarilla S4 (failed breakout) OR price < Donchian low
                if price <= stop_price or price < camarilla_s4_aligned[i] or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Camarilla R4 (failed breakout) OR price > Donchian high
                if price >= stop_price or price > camarilla_r4_aligned[i] or price >= donchian_high[i]:
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
        # Breakout continuation: price beyond Camarilla R4/S4 levels
        breakout_continuation_up = price > camarilla_r4_aligned[i]
        breakout_continuation_down = price < camarilla_s4_aligned[i]
        
        # Entry conditions: breakout in direction of Camarilla levels with volume confirmation
        long_setup = breakout_up and breakout_continuation_up and volume_confirmed
        short_setup = breakout_down and breakout_continuation_down and volume_confirmed
        
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