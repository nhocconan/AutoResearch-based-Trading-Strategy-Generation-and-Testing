#!/usr/bin/env python3
"""
Experiment #065: 12h Camarilla Pivot Breakout + 1d Volume Spike + Choppiness Filter

HYPOTHESIS: 12h price touching Camarilla pivot levels (L3/H3) from 1d timeframe,
confirmed by volume spike (>2x average) and non-choppy market (CHOP < 61.8),
captures institutional breakout/retest patterns. Discrete position sizing (0.25)
minimizes fee churn. Designed for 12-37 trades/year on 12h timeframe to avoid
overtrading while maintaining statistical significance. Works in both bull/bear
via regime filter and volume confirmation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss."""
    n = len(close)
    if n < 1:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values

def calculate_chop(high, low, close, period=14):
    """Choppiness Index: >61.8 = choppy/range, <38.2 = trending."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr_sum = np.zeros(n)
    for i in range(n):
        if i < period:
            atr_sum[i] = np.nan
            continue
        tr_sum = 0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        atr_sum[i] = tr_sum
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = np.zeros(n)
    for i in range(n):
        if np.isnan(atr_sum[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            chop[i] = np.nan
            continue
        if highest_high[i] == lowest_low[i]:
            chop[i] = 0
            continue
        log_val = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
        chop[i] = 100 * log_val
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots and chop (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels (based on previous day OHLC)
    # H4 = C + 1.5*(H-L), H3 = C + 1.0*(H-L), L3 = C - 1.0*(H-L), L4 = C - 1.5*(H-L)
    # where C = (H+L+O)/3 (typical price) or just (H+L+C)/3? Using (H+L+C)/3 as pivot
    # Standard Camarilla: Pivot = (H+L+C)/3
    # But we'll use typical price: (H+L+C)/3 for consistency
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_low_range = df_1d['high'] - df_1d['low']
    
    camarilla_h3 = typical_price + 1.0 * high_low_range  # H3 level
    camarilla_l3 = typical_price - 1.0 * high_low_range  # L3 level
    camarilla_h4 = typical_price + 1.5 * high_low_range  # H4 level (strong resistance)
    camarilla_l4 = typical_price - 1.5 * high_low_range  # L4 level (strong support)
    
    # Align HTF levels to LTF (12h)
    h3_1d = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    l3_1d = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    h4_1d = align_htf_to_ltf(prices, df_1d, camarilla_h4.values)
    l4_1d = align_htf_to_ltf(prices, df_1d, camarilla_l4.values)
    
    # === HTF: 1d Chop for regime filter (Call ONCE before loop) ===
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
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
        if (np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(h3_1d[i]) or np.isnan(l3_1d[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Choppiness Regime Filter (avoid breakouts in choppy markets) ---
        chop_ok = chop_1d_aligned[i] < 61.8  # Only trade when not excessively choppy
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Price Action at Camarilla Levels ---
        # Touch L3 (support) for long, H3 (resistance) for short
        # Using 0.1% tolerance for level touch
        tol = 0.001
        touch_l3 = abs(close[i] - l3_1d[i]) / close[i] < tol
        touch_h3 = abs(close[i] - h3_1d[i]) / close[i] < tol
        
        # --- Breakout beyond H4/L4 for momentum plays ---
        breakout_h4 = close[i] > h4_1d[i]
        breakdown_l4 = close[i] < l4_1d[i]
        
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
            
            # Exit conditions: opposite Camarilla touch or trend exhaustion
            min_hold = (i - entry_bar) >= 1  # Minimum 1 bar hold
            if min_hold:
                if position_side > 0:
                    # Exit long: touch H3 (resistance) OR break below L4
                    if touch_h3 or close[i] < l4_1d[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: touch L3 (support) OR break above H4
                    if touch_l3 or close[i] > h4_1d[i]:
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
        # Touch L3 (support) with volume confirmation and not choppy
        if touch_l3 and vol_ok and chop_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Touch H3 (resistance) with volume confirmation and not choppy
        elif touch_h3 and vol_ok and chop_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        # Alternative: Momentum breakouts beyond H4/L4 in strong trends
        elif breakout_h4 and vol_ok and chop_ok and close[i] > h3_1d[i]:
            # Only go long if already above H3 (confirming uptrend)
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        elif breakdown_l4 and vol_ok and chop_ok and close[i] < l3_1d[i]:
            # Only go short if already below L3 (confirming downtrend)
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals