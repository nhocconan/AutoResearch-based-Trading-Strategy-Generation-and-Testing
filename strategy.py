#!/usr/bin/env python3
"""
Experiment #214: 1h Donchian(20) breakout + 4h/1d Camarilla pivot + volume confirmation
HYPOTHESIS: Donchian breakouts on 1h timeframe aligned with higher-timeframe (4h/1d) Camarilla pivot levels (S3/R3 for mean reversion, S4/R4 for continuation) capture high-probability moves. Volume confirmation (>2.0x average) filters weak breakouts. ATR stoploss (2.0x) manages risk. Discrete position sizing (0.20) balances return and fee drag. Session filter (08-20 UTC) reduces noise trades. Target: 60-150 total trades over 4 years (15-37/year). Works in bull markets via breakout continuation at R4/S4 and in bear markets via mean reversion at R3/S3, with symmetry for longs/shorts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_214_1h_donchian20_4h_1d_camarilla_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === Pre-compute session hours (08-20 UTC) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels for 4h
    def calculate_camarilla(high_val, low_val, close_val):
        pt = (high_val + low_val + close_val) / 3.0
        rng = high_val - low_val
        r3 = pt + rng * 1.1 / 4
        r4 = pt + rng * 1.1 / 2
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    r3_4h = np.full(len(df_4h), np.nan)
    r4_4h = np.full(len(df_4h), np.nan)
    s3_4h = np.full(len(df_4h), np.nan)
    s4_4h = np.full(len(df_4h), np.nan)
    
    for i in range(len(df_4h)):
        if i >= 0:
            r3, r4, s3, s4 = calculate_camarilla(
                df_4h['high'].values[i],
                df_4h['low'].values[i],
                df_4h['close'].values[i]
            )
            r3_4h[i] = r3
            r4_4h[i] = r4
            s3_4h[i] = s3
            s4_4h[i] = s4
    
    # Align to 1h timeframe
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    r4_4h_aligned = align_htf_to_ltf(prices, df_4h, r4_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    s4_4h_aligned = align_htf_to_ltf(prices, df_4h, s4_4h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    r3_1d = np.full(len(df_1d), np.nan)
    r4_1d = np.full(len(df_1d), np.nan)
    s3_1d = np.full(len(df_1d), np.nan)
    s4_1d = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i >= 0:
            r3, r4, s3, s4 = calculate_camarilla(
                df_1d['high'].values[i],
                df_1d['low'].values[i],
                df_1d['close'].values[i]
            )
            r3_1d[i] = r3
            r4_1d[i] = r4
            s3_1d[i] = s3
            s4_1d[i] = s4
    
    # Align to 1h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r3_4h_aligned[i]) or np.isnan(r4_4h_aligned[i]) or
            np.isnan(s3_4h_aligned[i]) or np.isnan(s4_4h_aligned[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Camarilla Pivot Conditions (4h and 1d) ---
        near_r3_4h = abs(price - r3_4h_aligned[i]) / price < 0.005
        near_s3_4h = abs(price - s3_4h_aligned[i]) / price < 0.005
        break_r4_4h = price > r4_4h_aligned[i]
        break_s4_4h = price < s4_4h_aligned[i]
        
        near_r3_1d = abs(price - r3_1d_aligned[i]) / price < 0.005
        near_s3_1d = abs(price - s3_1d_aligned[i]) / price < 0.005
        break_r4_1d = price > r4_1d_aligned[i]
        break_s4_1d = price < s4_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 4h/1d S4 break with volume spike
                if (break_s4_4h or break_s4_1d) and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 4h/1d R4 break with volume spike
                if (break_r4_4h or break_r4_1d) and volume_spike:
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
        # Require volume spike + breakout conditions
        if volume_spike and in_session:
            # Long: breakout up AND (near R3 OR break R4) on either 4h or 1d
            long_condition = breakout_up and (
                (near_r3_4h or break_r4_4h) or 
                (near_r3_1d or break_r4_1d)
            )
            # Short: breakout down AND (near S3 OR break S4) on either 4h or 1d
            short_condition = breakout_down and (
                (near_s3_4h or break_s4_4h) or 
                (near_s3_1d or break_s4_1d)
            )
            
            if long_condition:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_condition:
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