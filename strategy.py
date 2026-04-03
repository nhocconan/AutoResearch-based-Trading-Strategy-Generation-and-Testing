#!/usr/bin/env python3
"""
Experiment #114: 4h/1d Donchian(20) Breakout + Volume Confirmation + Session Filter (1h timeframe)

HYPOTHESIS: 4h and 1d Donchian breakouts aligned with volume confirmation capture swing momentum
while minimizing whipsaw. 1h timeframe used only for precise entry timing. Session filter (08-20 UTC)
reduces noise trades. Discrete position sizing (0.20) and ATR trailing stop (2.0x) manage risk.
Targets 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to minimize fee drag.
Works in bull/bear markets by trading breakouts in direction of higher timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_1d_donchian_volume_session_1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for Donchian channels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    dh_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    dl_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    dh_4h_aligned = align_htf_to_ltf(prices, df_4h, dh_4h)
    dl_4h_aligned = align_htf_to_ltf(prices, df_4h, dl_4h)
    
    # === HTF: 1d data for Donchian channels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    dh_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    dl_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    dh_1d_aligned = align_htf_to_ltf(prices, df_1d, dh_1d)
    dl_1d_aligned = align_htf_to_ltf(prices, df_1d, dl_1d)
    
    # === 1h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dh_4h_aligned[i]) or np.isnan(dl_4h_aligned[i]) or
            np.isnan(dh_1d_aligned[i]) or np.isnan(dl_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Higher Timeframe Structure: Both 4h and 1d must agree ---
        # Bullish structure: price above both 4h and 1d lower Donchian
        bullish_structure = (close[i] > dl_4h_aligned[i]) and (close[i] > dl_1d_aligned[i])
        # Bearish structure: price below both 4h and 1d upper Donchian
        bearish_structure = (close[i] < dh_4h_aligned[i]) and (close[i] < dh_1d_aligned[i])
        
        # --- Price Channel Breakout (1h timeframe for entry timing) ---
        bullish_breakout = close[i] > dh_4h_aligned[i]  # Break above 4h upper channel
        bearish_breakout = close[i] < dl_4h_aligned[i]  # Break below 4h lower channel
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: opposite Donchian touch or structure breakdown
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches 4h lower Donchian OR structure turns bearish
                    if close[i] <= dl_4h_aligned[i] or not bullish_structure:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches 4h upper Donchian OR structure turns bullish
                    if close[i] >= dh_4h_aligned[i] or not bearish_structure:
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
        # Breakout above 4h upper Donchian with bullish 4h/1d structure and volume confirmation
        if bullish_breakout and bullish_structure and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below 4h lower Donchian with bearish 4h/1d structure and volume confirmation
        elif bearish_breakout and bearish_structure and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</script>