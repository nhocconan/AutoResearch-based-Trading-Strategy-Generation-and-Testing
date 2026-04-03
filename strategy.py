#!/usr/bin/env python3
"""
Experiment #035: 6h Weekly Pivot + 1d Volume Spike + 4h Trend Filter

HYPOTHESIS: Weekly pivot levels (from 1w data) provide significant support/resistance on 6h timeframe.
Combined with 1d volume spike confirmation and 4h EMA50 trend filter, this creates a strategy
that captures high-probability mean reversions at S1/S3/R1/R3 and breakouts at S4/R4.
Weekly pivots are more reliable than daily due to lower noise and institutional relevance.
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_pivot_vol_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot levels for each week
    pivot_points = np.full((len(df_1w), 5), np.nan)  # [S1, S2, S3, R1, R2, R3] but we'll use key levels
    if len(df_1w) >= 1:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Classic weekly pivot formulas
        pivot = (high_1w + low_1w + close_1w) / 3.0
        r1 = 2 * pivot - low_1w
        s1 = 2 * pivot - high_1w
        r2 = pivot + (high_1w - low_1w)
        s2 = pivot - (high_1w - low_1w)
        r3 = high_1w + 2 * (pivot - low_1w)
        s3 = low_1w - 2 * (high_1w - pivot)
        
        # Store key levels: S1, S3, R1, R3 (we'll also use S4/R4 as extremes)
        pivot_points[:, 0] = s1   # S1
        pivot_points[:, 1] = s3   # S3
        pivot_points[:, 2] = r1   # R1
        pivot_points[:, 3] = r3   # R3
        # For S4/R4 we'll use extensions
        pivot_points[:, 4] = s1 - (r1 - s1)  # S4 = S1 - (R1-S1)
        pivot_points[:, 5] = r3 + (r3 - s3)  # R4 = R3 + (R3-S3)
    
    # Align weekly pivot levels to 6h timeframe
    if len(df_1w) >= 1:
        s1_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 0])
        s3_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 1])
        r1_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 2])
        r3_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 3])
        s4_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 4])
        r4_aligned = align_htf_to_ltf(prices, df_1w, pivot_points[:, 5])
    else:
        s1_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s4_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (price > 4h EMA50 for long, < for short) ---
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly S4 (strong support) or R4 (strong resistance)
                if close[i] >= r4_aligned[i] or close[i] <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at weekly R4 (strong resistance) or S4 (strong support)
                if close[i] >= r4_aligned[i] or close[i] <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price at S3 (mean reversion) OR break above R3 with volume
        long_condition = (
            (close[i] <= s3_aligned[i] * 1.002 and price_above_4h_ema) or  # S3 mean reversion in uptrend
            (close[i] > r3_aligned[i] and volume_spike and price_above_4h_ema)  # Breakout with volume
        )
        
        # Short: Price at R3 (mean reversion) OR break below S3 with volume
        short_condition = (
            (close[i] >= r3_aligned[i] * 0.998 and price_below_4h_ema) or  # R3 mean reversion in downtrend
            (close[i] < s3_aligned[i] and volume_spike and price_below_4h_ema)  # Breakdown with volume
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals