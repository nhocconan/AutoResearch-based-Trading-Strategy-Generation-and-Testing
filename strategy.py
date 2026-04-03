#!/usr/bin/env python3
"""
Experiment #294: 1h Supertrend(10,3) + 4h/1d Donchian(20) trend filter + volume spike + session filter (08-20 UTC)
HYPOTHESIS: Supertrend catches momentum on 1h while 4h/1d Donchian channels ensure we only trade with the higher timeframe trend. Volume spike confirms institutional participation. Session filter avoids low-liquidity Asian session noise. Discrete sizing (0.20) minimizes fee drag. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_294_1h_supertrend10_3_4h1d_donchian20_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for Donchian channels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian(20)
    donch_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # 1d Donchian(20)
    donch_high_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    
    # === 1h Indicators: Supertrend(10,3) ===
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high + low) / 2 + 3 * atr
    basic_lb = (high + low) / 2 - 3 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(n)
    final_lb = np.zeros(n)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    for i in range(1, n):
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 for uptrend, -1 for downtrend
    supertrend[0] = final_ub[0]
    direction[0] = 1
    for i in range(1, n):
        if close[i] > final_ub[i-1]:
            direction[i] = 1
        elif close[i] < final_lb[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = final_lb[i]
        else:
            supertrend[i] = final_ub[i]
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC (pre-compute hours) ===
    # prices.index is already DatetimeIndex
    hours = prices.index.hour.values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # enough for Donchian(20) and Supertrend
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(donch_high_1d_aligned[i]) or np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(supertrend[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hour <= 20):
            if in_position:
                # Keep position but don't allow new entries outside session
                signals[i] = position_side * SIZE
                continue
            else:
                signals[i] = 0.0
                continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Conditions: Must align with BOTH 4h and 1d Donchian ---
        # Uptrend: price above both 4h and 1d Donchian lower bands
        # Downtrend: price below both 4h and 1d Donchian upper bands
        uptrend_4h = price > donch_low_4h_aligned[i]
        uptrend_1d = price > donch_low_1d_aligned[i]
        downtrend_4h = price < donch_high_4h_aligned[i]
        downtrend_1d = price < donch_high_1d_aligned[i]
        
        # --- Supertrend Entry Signals ---
        supertrend_long = direction[i] == 1 and price > supertrend[i]
        supertrend_short = direction[i] == -1 and price < supertrend[i]
        
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: Supertrend reversal or loss of 1h/4h/1d alignment
                if (direction[i] == -1 or  # Supertrend flipped
                    not (price > donch_low_4h_aligned[i] and price > donch_low_1d_aligned[i])):  # Lost uptrend alignment
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit: Supertrend reversal or loss of 1h/4h/1d alignment
                if (direction[i] == 1 or  # Supertrend flipped
                    not (price < donch_high_4h_aligned[i] and price < donch_high_1d_aligned[i])):  # Lost downtrend alignment
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Supertrend bullish AND price above both 4h and 1d Donchian lower bands
            if supertrend_long and uptrend_4h and uptrend_1d:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Supertrend bearish AND price below both 4h and 1d Donchian upper bands
            elif supertrend_short and downtrend_4h and downtrend_1d:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals