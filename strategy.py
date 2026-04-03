#!/usr/bin/env python3
"""
Experiment #251: 6h Camarilla Pivot + Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1-day HTF provide institutional support/resistance. 
We trade breakouts at R4/S4 with volume confirmation (>2.0x average) only when 6h price 
is above/below the 6h EMA50 (trend filter). In ranging markets (price near pivots), we avoid 
trades to reduce whipsaw. Uses ATR-based stoploss (2.5x) and minimum 6-bar holding period. 
Target: 75-150 trades over 4 years.
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
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for previous day
    # Camarilla: R4 = Close + 1.5*(High-Low), R3 = Close + 1.1*(High-Low)
    #            S3 = Close - 1.1*(High-Low), S4 = Close - 1.5*(High-Low)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    rng_1d = h_1d - l_1d
    r4_1d = c_1d + 1.5 * rng_1d
    r3_1d = c_1d + 1.1 * rng_1d
    s3_1d = c_1d - 1.1 * rng_1d
    s4_1d = c_1d - 1.5 * rng_1d
    
    # Align HTF levels to LTF (6h) with shift(1) for completed bars only
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 6h Indicators: EMA50 for trend filter ===
    ema50_6h = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
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
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for 6h EMA50 stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_6h[i]) or np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- 6h Trend Filter: Only trade when price is clearly above/below EMA50 ---
        price = close[i]
        ema50 = ema50_6h[i]
        # Require price to be at least 0.5*ATR away from EMA50 to avoid chop
        trend_filter = abs(price - ema50) > 0.5 * atr_14[i]
        is_uptrend = price > ema50 and trend_filter
        is_downtrend = price < ema50 and trend_filter
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Breakout Conditions at Camarilla R4/S4 levels ---
        breakout_up = high[i] > r4_1d_aligned[i-1]  # Break above R4
        breakout_down = low[i] < s4_1d_aligned[i-1]  # Break below S4
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on break below R3 (take profit at first support)
                if low[i] < r3_1d_aligned[i]:
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
                # Exit on break above S3 (take profit at first resistance)
                if high[i] > s3_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 6 bars to reduce churn
            if bars_since_entry < 6:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above R4 with volume spike in uptrend
        if is_uptrend and breakout_up and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Break below S4 with volume spike in downtrend
        elif is_downtrend and breakout_down and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals