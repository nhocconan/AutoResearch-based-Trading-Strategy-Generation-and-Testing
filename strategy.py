#!/usr/bin/env python3
"""
Experiment #048: 12h Camarilla Pivot Breakout + 1w Volume Spike + ATR Stoploss

HYPOTHESIS: 12h Camarilla pivot levels (derived from 1d OHLC) act as strong 
support/resistance zones. A breakout above R4 or below S4 with 1w volume 
confirmation (2.0x average volume) captures institutional interest. 
Uses ATR-based stoploss (2.5x) for risk management. Designed for 15-25 
trades/year to minimize fee drag while maintaining statistical significance. 
Discrete position sizing (0.25) reduces churn from minor signal fluctuations.
Works in both bull (breakouts continue trend) and bear (breakdowns accelerate) 
markets due to volume confirmation filtering false breakouts.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_1w_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w Volume for confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # === 12h Indicators (from 1d data for Camarilla) ===
    df_1d = get_htf_data(prices, '1d')
    # Camarilla levels from previous 1d bar
    camarilla_H4 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    camarilla_L4 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 2
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # ATR for stoploss
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        if (np.isnan(atr_14[i]) or np.isnan(camarilla_H4_aligned[i]) or 
            np.isnan(camarilla_L4_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Camarilla Levels (from previous 1d bar) ---
        H4 = camarilla_H4_aligned[i]
        L4 = camarilla_L4_aligned[i]
        
        # --- Volume Confirmation (1w average volume) ---
        vol_ok = volume[i] > vol_ma_20_1w_aligned[i] * 2.0 if vol_ma_20_1w_aligned[i] > 1e-10 else False
        
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
            
            # Exit conditions: price returns to Camarilla H3/L3 levels
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1d)
            if min_hold:
                camarilla_H3 = df_1d['close'].values[max(0, i//16)] + (df_1d['high'].values[max(0, i//16)] - df_1d['low'].values[max(0, i//16)]) * 1.1/4
                camarilla_L3 = df_1d['close'].values[max(0, i//16)] - (df_1d['high'].values[max(0, i//16)] - df_1d['low'].values[max(0, i//16)]) * 1.1/4
                # Align H3/L3 to LTF (using same 1d data)
                camarilla_H3_series = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1/4
                camarilla_L3_series = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1/4
                camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3_series)
                camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3_series)
                
                if position_side > 0:
                    # Exit long: price returns to H3 level
                    if close[i] <= camarilla_H3_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price returns to L3 level
                    if close[i] >= camarilla_L3_aligned[i]:
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
        # Breakout above Camarilla H4 with 1w volume confirmation
        if close[i] > H4 and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakdown below Camarilla L4 with 1w volume confirmation
        elif close[i] < L4 and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals