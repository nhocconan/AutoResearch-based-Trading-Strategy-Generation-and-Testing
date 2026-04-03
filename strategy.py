#!/usr/bin/env python3
"""
Experiment #052: 12h Camarilla Pivot Levels + 1d Volume Spike + Chop Regime Filter

HYPOTHESIS: Camarilla pivot levels (L3/H3) from daily timeframe act as intraday support/resistance.
In 12h timeframe, price approaching these levels with volume confirmation (>2x average) and
choppiness regime filter (CHOP > 50 for mean-reversion) provides high-probability reversal trades.
ATR-based stoploss (2.0x) manages risk. Designed for 15-25 trades/year to minimize fee drag
while capturing mean-reversion moves in both bull and bear markets. Discrete position sizing 
(0.25) reduces churn.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_1d_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def calculate_chop(high, low, close, period=14):
    """Choppiness Index: measures whether market is choppy (ranging) or trending."""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    hh = pd.Series(high).rolling(window=period, min_periods=period).max()
    ll = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    # Handle division by zero when hh == ll
    chop = np.where((hh - ll) == 0, 50.0, chop)
    return chop.values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d Camarilla Pivot Levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Typical price for pivot calculation
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels
    pivot = typical_price
    range_hl = df_1d['high'] - df_1d['low']
    # Camarilla L3 and H3 (most important for reversals)
    camarilla_l3 = pivot - (range_hl * 1.1 / 4)
    camarilla_h3 = pivot + (range_hl * 1.1 / 4)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3.values)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3.values)
    
    # === 12h Indicators ===
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    # Chop regime filter
    chop = calculate_chop(high, low, close, period=14)
    # ATR for stoploss
    tr = np.zeros(n)
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Camarilla Levels ---
        l3 = camarilla_l3_aligned[i]
        h3 = camarilla_h3_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
        # --- Chop Regime Filter (CHOP > 50 = ranging/mean-reversion favorable) ---
        chop_ok = chop[i] > 50.0
        
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
            
            # Exit conditions: price reaches opposite Camarilla level or chop regime breaks down
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1 day)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches H3 OR chop < 40 (trending)
                    if close[i] >= h3 or chop[i] < 40.0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches L3 OR chop < 40 (trending)
                    if close[i] <= l3 or chop[i] < 40.0:
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
        # Price near L3 (within 0.5%) with volume confirmation AND chop regime favorable
        if low[i] <= l3 * 1.005 and close[i] > l3 and vol_ok and chop_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Price near H3 (within 0.5%) with volume confirmation AND chop regime favorable
        elif high[i] >= h3 * 0.995 and close[i] < h3 and vol_ok and chop_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals