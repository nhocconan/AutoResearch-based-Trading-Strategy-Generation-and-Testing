#!/usr/bin/env python3
"""
Experiment #235: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 6h aligned with weekly Camarilla pivot structure (R3/S3 for fade, R4/S4 for breakout) capture institutional interest. Volume confirmation (>1.8x average) filters weak moves. ATR stoploss (2.5x) manages risk in volatile 6h candles. Discrete position sizing (0.25) balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year). Works in bull markets via R4/S4 breakout continuation and in bear markets via R3/S3 mean reversion, with symmetry for longs/shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_235_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    # Camarilla: P = (H+L+C)/3, Range = H-L
    # R4 = C + Range * 1.1/2, R3 = C + Range * 1.1/4, S3 = C - Range * 1.1/4, S4 = C - Range * 1.1/2
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_range = df_1w['high'] - df_1w['low']
    camarilla_pivot = typical_price
    camarilla_r4 = camarilla_pivot + weekly_range * 1.1 / 2
    camarilla_r3 = camarilla_pivot + weekly_range * 1.1 / 4
    camarilla_s3 = camarilla_pivot - weekly_range * 1.1 / 4
    camarilla_s4 = camarilla_pivot - weekly_range * 1.1 / 2
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot.values)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4.values)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Camarilla Pivot Conditions ---
        # Near R3/S3: within 0.5% of level (fade zone)
        near_r3 = abs(price - r3_aligned[i]) / r3_aligned[i] < 0.005
        near_s3 = abs(price - s3_aligned[i]) / s3_aligned[i] < 0.005
        # Beyond R4/S4: breakout level (continuation zone)
        beyond_r4 = price > r4_aligned[i]
        beyond_s4 = price < s4_aligned[i]
        
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
                # Exit conditions: fade at R3 or continuation break of S4
                if (near_r3 and volume_spike) or (breakout_down and volume_spike and beyond_s4):
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
                # Exit conditions: fade at S3 or continuation break of R4
                if (near_s3 and volume_spike) or (breakout_up and volume_spike and beyond_r4):
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
        # Require volume spike + breakout conditions + Camarilla alignment
        if volume_spike:
            # Long: breakout up AND (beyond R4 for continuation OR near S3 for fade)
            if breakout_up and (beyond_r4 or near_s3):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND (beyond S4 for continuation OR near R3 for fade)
            elif breakout_down and (beyond_s4 or near_r3):
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