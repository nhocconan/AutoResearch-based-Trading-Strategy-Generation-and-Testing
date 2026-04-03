#!/usr/bin/env python3
"""
Experiment #097: 4h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Spike + Chop Regime

HYPOTHESIS: 4h Donchian breakouts aligned with 1d Camarilla pivot levels (S3/S4 for shorts, R3/R4 for longs)
capture institutional order flow at key support/resistance. Volume confirmation (2.0x average) ensures follow-through.
Choppiness regime filter (CHOP > 61.8) avoids trending markets where pivots fail. Designed for 20-50 trades/year on 4h
timeframe to minimize fee drag while maintaining statistical significance. Uses discrete position sizing (0.25) to reduce churn.
Works in both bull/bear markets by trading mean-reversion at extreme pivot levels during ranging conditions.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index: measures whether market is choppy (ranging) or trending"""
    n = len(close)
    if n < period:
        return np.full(n, 50.0)
    
    atr_sum = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # True Range sum over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.where(range_hl > 1e-10, 
                    100 * np.log10(atr_sum / range_hl) / np.log10(period), 
                    50.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Camarilla: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 4h timeframe (shifted by 1 for completed bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 4h Indicators ===
    # ATR for volatility and stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian channels for breakout confirmation
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index for regime filter (choppy = ranging = good for pivot mean reversion)
    chop = calculate_choppiness(high, low, close, 14)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness > 61.8 = ranging market (good for pivot mean reversion) ---
        chop_ok = chop[i] > 61.8
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Price near Camarilla Extreme Levels ---
        # Long: price near S3/S4 (strong support) with bullish bias
        near_s4 = abs(close[i] - s4_aligned[i]) < (0.5 * atr_14[i])  # within 0.5 ATR of S4
        near_s3 = abs(close[i] - s3_aligned[i]) < (0.5 * atr_14[i])  # within 0.5 ATR of S3
        
        # Short: price near R3/R4 (strong resistance) with bearish bias
        near_r4 = abs(close[i] - r4_aligned[i]) < (0.5 * atr_14[i])  # within 0.5 ATR of R4
        near_r3 = abs(close[i] - r3_aligned[i]) < (0.5 * atr_14[i])  # within 0.5 ATR of R3
        
        # --- Donchian Breakout Confirmation ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
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
            
            # Exit conditions: price reaches opposite pivot level or Donchian touch
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price reaches R3/R4 (resistance) OR touches lower Donchian
                    if close[i] >= r3_aligned[i] or close[i] <= dc_lower_20[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price reaches S3/S4 (support) OR touches upper Donchian
                    if close[i] <= s3_aligned[i] or close[i] >= dc_upper_20[i]:
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
        # Price near S3/S4 support AND bullish Donchian breakout AND volume confirmation AND choppy market
        if ((near_s4 or near_s3) and bullish_breakout and vol_ok and chop_ok):
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Price near R3/R4 resistance AND bearish Donchian breakout AND volume confirmation AND choppy market
        elif ((near_r4 or near_r3) and bearish_breakout and vol_ok and chop_ok):
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals