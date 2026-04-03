#!/usr/bin/env python3
"""
Experiment #035: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels (breakout at R4/S4 for continuation, fade at R3/S3) with volume confirmation (>1.5x) captures institutional flow. Weekly trend filter (price above/below 1w EMA20) ensures direction alignment with higher timeframe. Discrete sizing (0.25) and ATR(14) stoploss (2.5x) manages risk. Target: 100-200 total trades over 4 years (25-50/year) for statistical validity and low fee drag. Works in bull (R4 breakouts with weekly uptrend) and bear (S4 breakdowns with weekly downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_035_6h_donchian20_1d_camarilla_1w_trend_v1"
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
    # Based on previous day's high, low, close
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    pivot = (h_1d + l_1d + c_1d) / 3.0
    range_hl = h_1d - l_1d
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 2.0)
    r4 = pivot + (range_hl * 1.1)
    s3 = pivot - (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === HTF: 1w data for trend filter (EMA20) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(20) on 1w close
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema_1w_6h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Trend Filter ---
        weekly_uptrend = price > ema_1w_6h[i]
        weekly_downtrend = price < ema_1w_6h[i]
        
        # --- Camarilla Logic ---
        # Fade at R3/S3 (mean reversion)
        fade_long = price <= s3_6h[i] and price >= s4_6h[i]  # Near strong support
        fade_short = price >= r3_6h[i] and price <= r4_6h[i]  # Near strong resistance
        
        # Breakout continuation at R4/S4 (institutional break)
        breakout_long = price >= r4_6h[i]  # Break above strong resistance
        breakout_short = price <= s4_6h[i]  # Break below strong support
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~3d on 6h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up AND weekly uptrend AND (Camarilla breakout at R4 OR fade from S3)
            if breakout_up and weekly_uptrend and (breakout_long or fade_long):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down AND weekly downtrend AND (Camarilla breakout at S4 OR fade from R3)
            elif breakout_down and weekly_downtrend and (breakout_short or fade_short):
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