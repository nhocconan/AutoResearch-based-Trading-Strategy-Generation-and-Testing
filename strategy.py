#!/usr/bin/env python3
"""
Experiment #3551: 6h Volume Spike + 1d Weekly Pivot Fade
HYPOTHESIS: Fade extreme weekly pivot levels (R4/S4) on 6h with volume spike confirmation captures mean reversion in both bull and bear markets. 
Weekly R4/S4 act as institutional barriers where price often reverses. Volume spike confirms exhaustion. 
Position size 0.25. Target: 100-200 total trades over 4 years (25-50/year).
Uses 1d for pivot calculation (R4/S4 levels), 6h only for entry timing and risk management.
Works in bull (fade from R4) and bear (fade from S4) via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3551_6h_volume_spike_1d_pivot_fade_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (using prior week's data)
    lookback_week = 5
    prior_week_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
    prior_week_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
    prior_week_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
    
    # Weekly pivot formula: P = (H + L + C) / 3
    weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
    # Weekly R1/S1
    r1 = 2 * weekly_pivot - prior_week_low
    s1 = 2 * weekly_pivot - prior_week_high
    # Weekly R2/S2
    r2 = weekly_pivot + (prior_week_high - prior_week_low)
    s2 = weekly_pivot - (prior_week_high - prior_week_low)
    # Weekly R3/S3
    r3 = weekly_pivot + 2 * (prior_week_high - prior_week_low)
    s3 = weekly_pivot - 2 * (prior_week_high - prior_week_low)
    # Weekly R4/S4 (extreme levels)
    r4 = weekly_pivot + 3 * (prior_week_high - prior_week_low)
    s4 = weekly_pivot - 3 * (prior_week_high - prior_week_low)
    
    # Align all levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(50, lookback_week + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price moves 2*ATR against position (stoploss)
            if position_side > 0:  # Long
                if price < entry_price - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly pivot (mean reversion target)
                elif price >= weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > entry_price + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches weekly pivot (mean reversion target)
                elif price <= weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Fade at extreme weekly levels: R4/S4
            # Short at R4 with expectation of reversion to pivot
            if price >= r4_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            # Long at S4 with expectation of reversion to pivot
            elif price <= s4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals