#!/usr/bin/env python3
"""
Experiment #5319: 6h Donchian(20) breakout + 12h Supertrend + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian breakouts aligned with 12h Supertrend direction 
and volume > 2x average capture strong momentum moves with controlled frequency. 
Supertrend (ATR=10, mult=3.0) filters for trend direction, reducing false breakouts. 
Long when price breaks above Donchian upper band with volume confirmation and 
Supertrend uptrend; short when breaks below lower band with volume and downtrend. 
Uses discrete position sizing (0.25) and ATR-based trailing stop (3.0*ATR) to limit drawdown. 
Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5319_6h_donchian20_12h_supertrend_vol_v1"
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
    if len(df_12h) >= 2:
        # Supertrend calculation: ATR(10), multiplier 3.0
        atr_period = 10
        multiplier = 3.0
        
        # True Range
        tr1 = df_12h['high'] - df_12h['low']
        tr2 = np.abs(df_12h['high'] - df_12h['close'].shift(1))
        tr3 = np.abs(df_12h['low'] - df_12h['close'].shift(1))
        tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_12h = tr_12h.rolling(window=atr_period, min_periods=atr_period).mean()
        
        # Basic Upper and Lower Bands
        hl2 = (df_12h['high'] + df_12h['low']) / 2
        basic_ub = hl2 + (multiplier * atr_12h)
        basic_lb = hl2 - (multiplier * atr_12h)
        
        # Final Upper and Lower Bands
        final_ub = basic_ub.copy()
        final_lb = basic_lb.copy()
        for i in range(1, len(df_12h)):
            if basic_ub[i] < final_ub[i-1] or df_12h['close'].iloc[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or df_12h['close'].iloc[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
        
        # Supertrend direction: 1 for uptrend, -1 for downtrend
        supertrend = np.ones(len(df_12h)) * -1  # Initialize as downtrend
        for i in range(1, len(df_12h)):
            if df_12h['close'].iloc[i] > final_ub[i-1]:
                supertrend[i] = 1
            elif df_12h['close'].iloc[i] < final_lb[i-1]:
                supertrend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
                if supertrend[i] == 1 and final_lb[i] < final_lb[i-1]:
                    final_lb[i] = final_lb[i-1]
                if supertrend[i] == -1 and final_ub[i] > final_ub[i-1]:
                    final_ub[i] = final_ub[i-1]
        
        # Align to LTF (6h) with shift(1) for completed bars only
        supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    else:
        supertrend_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 2)  # Donchian, volume avg, ATR, Supertrend warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (21-23 UTC) ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(supertrend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 3.0 * ATR below highest since entry
                stop_price = highest_since_entry - 3.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks below Donchian lower band (failed breakout)
                # 3. Supertrend turns down (trend reversal)
                if price <= stop_price or price <= donchian_low[i] or supertrend_aligned[i] == -1:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 3.0 * ATR above lowest since entry
                stop_price = lowest_since_entry + 3.0 * atr[i]
                # Exit conditions:
                # 1. Stoploss hit
                # 2. Price breaks above Donchian upper band (failed breakout)
                # 3. Supertrend turns up (trend reversal)
                if price >= stop_price or price >= donchian_high[i] or supertrend_aligned[i] == 1:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Donchian breakout conditions
        breakout_up = price > donchian_high[i-1]  # Break above previous period's high
        breakout_down = price < donchian_low[i-1]  # Break below previous period's low
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Supertrend direction filter
        supertrend_up = supertrend_aligned[i-1] == 1
        supertrend_down = supertrend_aligned[i-1] == -1
        
        # Entry conditions: breakout + volume + Supertrend alignment
        if breakout_up and volume_confirmed and supertrend_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and supertrend_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals