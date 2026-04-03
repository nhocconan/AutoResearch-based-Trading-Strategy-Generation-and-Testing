#!/usr/bin/env python3
"""
Experiment #251: 6h Camarilla Pivot + Volume Spike + Regime Filter (ADX/Chop)
HYPOTHESIS: Camarilla pivot levels (R3/S3, R4/S4) from 1d act as institutional support/resistance. 
Entries on 6h: 
  - Long when price crosses above R3 with volume spike (>2x avg) AND choppy regime (CHOP > 61.8 = range)
  - Short when price crosses below S3 with volume spike AND choppy regime
  - Exit when price reaches R4/S4 (full mean reversion target) or opposite S3/R3
  - In trending regime (CHOP < 38.2), only trade breakouts of R4/S4 with volume (continuation)
Uses discrete sizing (0.25) to minimize fee drag. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_251_6h_camarilla_pivot_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivots (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # Camarilla: based on prior day's range, not weekly
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: Choppiness Index (CHOP) for regime detection ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_14_series = pd.Series(tr_6h)
    sum_atr_14 = atr_14_series.rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = np.zeros(n)
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    chop[14:] = 100 * np.log10(sum_atr_14[14:] / hl_range[14:]) / np.log10(14)
    chop[:14] = 50.0  # neutral
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 40  # Enough for 20-period volume MA and 14-period indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Detection ---
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at R4 (full mean reversion target) in ranging
                # or trail stop at 2*ATR profit in trending
                if is_ranging and high[i] >= r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if is_trending and high[i] < entry_price + 2.0 * atr_14[i]:
                    # Trail stop: exit if price drops 2*ATR from peak
                    peak_since_entry = np.maximum.accumulate(high[bars_since_entry-max(0, i-20):i+1])[-1] if i > bars_since_entry else entry_price
                    if high[i] < peak_since_entry - 2.0 * atr_14[i]:
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at S4 (full mean reversion target) in ranging
                # or trail stop at 2*ATR profit in trending
                if is_ranging and low[i] <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                if is_trending and low[i] > entry_price - 2.0 * atr_14[i]:
                    # Trail stop: exit if price rises 2*ATR from trough
                    trough_since_entry = np.minimum.accumulate(low[bars_since_entry-max(0, i-20):i+1])[-1] if i > bars_since_entry else entry_price
                    if low[i] > trough_since_entry + 2.0 * atr_14[i]:
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
        
        # --- New Position Entry Logic (Only if Flat) ---
        if volume_spike:
            # Ranging market: mean reversion at R3/S3
            if is_ranging:
                # Long: price crosses above R3 (support hold)
                if close[i-1] <= r3_aligned[i-1] and price > r3_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price crosses below S3 (resistance hold)
                elif close[i-1] >= s3_aligned[i-1] and price < s3_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            # Trending market: breakout continuation at R4/S4
            elif is_trending:
                # Long: breakout above R4 with volume
                if close[i-1] <= r4_aligned[i-1] and price > r4_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: breakdown below S4 with volume
                elif close[i-1] >= s4_aligned[i-1] and price < s4_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals