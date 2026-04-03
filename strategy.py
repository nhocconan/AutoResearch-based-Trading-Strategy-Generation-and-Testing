#!/usr/bin/env python3
"""
Experiment #195: 6h Camarilla Pivot + Volume Spike + Weekly Trend Filter
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 1d timeframe combined with volume spikes (>2x average) and weekly trend filter (price above/below weekly VWAP) capture high-probability reversals and continuations with optimal trade frequency for 6h. Weekly trend ensures alignment with higher timeframe momentum. Discrete position sizing (0.25) and ATR stoploss (2.5x) manage risk. Target: 75-150 total trades over 4 years. Works in bull/bear via mean reversion at extremes and breakout continuation with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_195_6h_camarilla_pivot_volume_weekly_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(high, low, close):
        pt = (high + low + close) / 3.0
        rng = high - low
        r3 = pt + rng * 1.1 / 4
        r4 = pt + rng * 1.1 / 2
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4
    
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
    
    # Align to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === HTF: 1w data for weekly trend filter (VWAP) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly VWAP
    typical_price_1w = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vol_1w = df_1w['volume']
    vwap_1w = (typical_price_1w * vol_1w).cumsum() / vol_1w.cumsum()
    vwap_1w_values = vwap_1w.values
    
    # Align weekly VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1d, vwap_1w_values)  # Use 1d as bridge for alignment
    
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
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(vwap_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Weekly Trend Filter ---
        weekly_uptrend = price > vwap_1w_aligned[i]
        weekly_downtrend = price < vwap_1w_aligned[i]
        
        # --- Camarilla Pivot Conditions ---
        near_r3 = abs(price - r3_1d_aligned[i]) / price < 0.003
        near_s3 = abs(price - s3_1d_aligned[i]) / price < 0.003
        break_r4 = price > r4_1d_aligned[i]
        break_s4 = price < s4_1d_aligned[i]
        
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
                # Take profit at R4 with volume spike
                if break_r4 and volume_spike:
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
                # Take profit at S4 with volume spike
                if break_s4 and volume_spike:
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
        # Require volume spike
        if volume_spike:
            # Long: near S3 (mean reversion) in weekly uptrend OR break S4 (continuation) in weekly downtrend
            if (near_s3 and weekly_uptrend) or (break_s4 and weekly_downtrend):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: near R3 (mean reversion) in weekly downtrend OR break R4 (continuation) in weekly uptrend
            elif (near_r3 and weekly_downtrend) or (break_r4 and weekly_uptrend):
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