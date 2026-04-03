#!/usr/bin/env python3
"""
Experiment #118: 1d Donchian(20) breakout + 1w Camarilla pivot + volume confirmation + ATR stoploss
HYPOTHESIS: Daily Donchian breakouts aligned with weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for continuation) capture high-probability moves across market regimes. Volume confirmation (>1.8x average) filters weak breakouts. ATR stoploss (2.5x) manages risk. Discrete position sizing (0.30) balances return and drawdown. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_118_1d_donchian20_1w_camarilla_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels for 1w
    def calculate_camarilla(high, low, close):
        pt = (high + low + close) / 3.0
        rng = high - low
        r3 = pt + rng * 1.1 / 4
        r4 = pt + rng * 1.1 / 2
        s3 = pt - rng * 1.1 / 4
        s4 = pt - rng * 1.1 / 2
        return r3, r4, s3, s4
    
    r3_1w = np.full(len(df_1w), np.nan)
    r4_1w = np.full(len(df_1w), np.nan)
    s3_1w = np.full(len(df_1w), np.nan)
    s4_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        if i >= 0:
            r3, r4, s3, s4 = calculate_camarilla(
                df_1w['high'].values[i],
                df_1w['low'].values[i],
                df_1w['close'].values[i]
            )
            r3_1w[i] = r3
            r4_1w[i] = r4
            s3_1w[i] = s3
            s4_1w[i] = s4
    
    # Align to 1d timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30
    
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
            np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Camarilla Pivot Conditions ---
        near_r3 = abs(price - r3_1w_aligned[i]) / price < 0.005
        near_s3 = abs(price - s3_1w_aligned[i]) / price < 0.005
        break_r4 = price > r4_1w_aligned[i]
        break_s4 = price < s4_1w_aligned[i]
        
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
                if break_s4 and volume_spike:
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
                if break_r4 and volume_spike:
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
        if (breakout_up and volume_spike and (near_r3 or break_r4)) or (break_r4 and volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif (breakout_down and volume_spike and (near_s3 or break_s4)) or (break_s4 and volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals