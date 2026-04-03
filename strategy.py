#!/usr/bin/env python3
"""
Experiment #279: 6h Camarilla Pivot + 12h Volume Spike + 1d Trend Filter

HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) on 6h timeframe,
filtered by 12h volume spikes (>2.0x average) and 1d EMA50 trend direction, capture high-probability
trades with minimal false signals. The 12h volume spike confirms institutional participation,
while the 1d EMA50 ensures alignment with the dominant trend. This strategy targets 12-37 trades/year
(50-150 total over 4 years) to minimize fee drag while maintaining statistical significance.
Works in bull markets (breakouts at R4/S4 with volume) and bear markets (mean reversion at R3/S3).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_279_6h_camarilla_12h_volume_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume MA (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === HTF: 1d data for EMA50 trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Calculate pivot from previous bar's OHLC
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_r3[i] = pivot + (range_val * 1.1 / 4.0)
        camarilla_s3[i] = pivot - (range_val * 1.1 / 4.0)
        camarilla_r4[i] = pivot + (range_val * 1.1 / 2.0)
        camarilla_s4[i] = pivot - (range_val * 1.1 / 2.0)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 60  # Ensure enough data for HTF indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- 12h Volume Spike Confirmation: Require volume > 2.0x average ---
        volume_spike = volume[i] > (2.0 * vol_ma_12h_aligned[i])
        
        # --- 1d EMA50 Trend Filter ---
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Camarilla Conditions ---
        # Mean reversion at R3/S3 (fade extreme moves)
        mean_reversion_long = close[i] <= camarilla_s3[i]
        mean_reversion_short = close[i] >= camarilla_r3[i]
        
        # Breakout continuation at R4/S4 (strong momentum)
        breakout_long = close[i] >= camarilla_r4[i]
        breakout_short = close[i] <= camarilla_s4[i]
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Mean reversion exit: price returns to pivot zone
            if position_side > 0 and close[i] >= camarilla_s3[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            if position_side < 0 and close[i] <= camarilla_r3[i]:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions:
        # 1. Mean reversion: price at S3 + volume spike + above 1d EMA50
        # 2. Breakout: price at R4 + volume spike + above 1d EMA50
        long_condition = (
            (mean_reversion_long or breakout_long) and 
            volume_spike and 
            price_above_ema
        )
        
        # Short conditions:
        # 1. Mean reversion: price at R3 + volume spike + below 1d EMA50
        # 2. Breakout: price at S4 + volume spike + below 1d EMA50
        short_condition = (
            (mean_reversion_short or breakout_short) and 
            volume_spike and 
            price_below_ema
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
    
    return signals