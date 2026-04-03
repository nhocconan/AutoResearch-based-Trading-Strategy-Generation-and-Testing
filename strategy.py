#!/usr/bin/env python3
"""
Experiment #092: 12h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Spike

HYPOTHESIS: 12h Donchian breakouts confirmed by 1d Camarilla pivot levels (L3/H3) and volume spikes capture medium-term trends with high confluence.
The 12h timeframe reduces noise while providing sufficient trades (target: 50-150 over 4 years). Volume confirmation ensures institutional participation.
Uses ATR-based stoploss for risk control. Works in bull/bear markets by trading breakouts in direction aligned with daily pivot bias.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    pivot = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    range_ = df_1d['high'].values - df_1d['low'].values
    # Camarilla levels: H3, L3 (most significant for intraday reversals/breakouts)
    camarilla_h3 = pivot + (range_ * 1.1 / 4)
    camarilla_l3 = pivot - (range_ * 1.1 / 4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 12h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Camarilla Pivot Filter ---
        # For longs: price should be above L3 (bullish bias)
        # For shorts: price should be below H3 (bearish bias)
        camarilla_long_filter = close[i] > camarilla_l3_aligned[i]
        camarilla_short_filter = close[i] < camarilla_h3_aligned[i]
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: opposite Donchian touch or Camarilla level reversal
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1day)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below Camarilla L3
                    if close[i] <= dc_lower_20[i] or close[i] < camarilla_l3_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above Camarilla H3
                    if close[i] >= dc_upper_20[i] or close[i] > camarilla_h3_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with volume confirmation and Camarilla bullish bias
        if bullish_breakout and vol_ok and camarilla_long_filter:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and Camarilla bearish bias
        elif bearish_breakout and vol_ok and camarilla_short_filter:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals