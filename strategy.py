#!/usr/bin/env python3
"""
Experiment #235: 6h Camarilla pivot + 1d trend filter + volume spike
HYPOTHESIS: On 6h timeframe, Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d EMA50 trend filter and volume confirmation (>1.5x average) captures high-probability reversals and continuations. In bull markets, longs at S3/S4 with trend; in bear markets, shorts at R3/R4 with trend. Uses discrete position sizing (0.25) to manage fee drag and drawdown. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_235_6h_camarilla_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === HTF: 1w data for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Weekly OHLC for pivot calculation
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + Range * 1.1
    # S3 = Pivot - Range * 1.1
    # R4 = Pivot + Range * 1.5
    # S4 = Pivot - Range * 1.5
    pivot_1w = (weekly_high + weekly_low + weekly_close) / 3.0
    range_1w = weekly_high - weekly_low
    r3_1w = pivot_1w + range_1w * 1.1
    s3_1w = pivot_1w - range_1w * 1.1
    r4_1w = pivot_1w + range_1w * 1.5
    s4_1w = pivot_1w - range_1w * 1.5
    
    # Align weekly pivot levels to 6h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
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
    
    warmup = 60  # Enough for weekly pivot and EMA50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s4_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- 1d Trend Condition ---
        trend_up = price > ema_1d_aligned[i]
        trend_down = price < ema_1d_aligned[i]
        
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
                # Take profit at opposite S3/R3 level
                if price > s3_1w_aligned[i] and position_side > 0:
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
                # Take profit at opposite R3/S3 level
                if price < r3_1w_aligned[i] and position_side < 0:
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
        # Mean reversion at S3/R3 with trend
        if volume_spike:
            # Long: price at or below S3 AND uptrend
            if price <= s3_1w_aligned[i] and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price at or above R3 AND downtrend
            elif price >= r3_1w_aligned[i] and trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # Breakout continuation at S4/R4 with trend
            elif price < s4_1w_aligned[i] and trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            elif price > r4_1w_aligned[i] and trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals