#!/usr/bin/env python3
"""
Experiment #142: 12h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: 12h Donchian breakouts aligned with 1d volume confirmation and
choppiness regime filter capture swing momentum while avoiding whipsaw in range
markets. The 12h timeframe reduces trade frequency to minimize fee drag, while
volume spike ensures institutional participation. Chop filter (>61.8) avoids false
breakouts in ranging markets. Discrete position sizing (0.25) and ATR trailing stop
(2.5x) manage risk. Targets 12-37 trades/year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Average True Range"""
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_chop(high, low, close, period):
    """Choppiness Index: measures whether market is trending or ranging"""
    atr_sum = pd.Series(calculate_atr(high, low, close, 1)).rolling(window=period, min_periods=period).sum()
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    return chop.values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    chop = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, 14)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20_12h[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- 12h Volume Confirmation ---
        vol_ok_12h = volume[i] > vol_ma_20_12h[i] * 1.5 if vol_ma_20_12h[i] > 1e-10 else False
        
        # --- 1d Volume Spike Confirmation ---
        # Get the 1d volume MA for current 12h bar (aligned)
        vol_spike_ok = vol_ma_20[i] > 0 and volume[i] > vol_ma_20[i] * 2.0  # 2x daily volume average
        
        # --- Chop Regime Filter: Only trade when market is trending (CHOP < 38.2) ---
        trending_market = chop_aligned[i] < 38.2
        
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
            
            # Exit conditions: opposite Donchian touch or chop becomes too high (ranging)
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~24h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR chop > 61.8 (ranging)
                    if close[i] <= dc_lower_20[i] or chop_aligned[i] > 61.8:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR chop > 61.8 (ranging)
                    if close[i] >= dc_upper_20[i] or chop_aligned[i] > 61.8:
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
        # Breakout above upper Donchian with volume confirmation (both timeframes) and trending market
        if bullish_breakout and vol_ok_12h and vol_spike_ok and trending_market:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation (both timeframes) and trending market
        elif bearish_breakout and vol_ok_12h and vol_spike_ok and trending_market:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals