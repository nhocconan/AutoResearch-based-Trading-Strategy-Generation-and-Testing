#!/usr/bin/env python3
"""
Experiment #235: 6h Weekly Pivot Donchian Breakout Strategy

HYPOTHESIS: Weekly pivot points (PP, R1-4, S1-4) combined with 6h Donchian(20) breakouts and volume confirmation capture institutional-level breakouts and reversals. In bull/bear markets, we trade breakouts in the direction of weekly pivot bias. In ranging markets, we fade extreme weekly pivot levels (R3/S3, R4/S4) with volume exhaustion signals. Weekly pivots provide structure that works across regimes, while 6h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag in BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_235_6h_weekly_pivot_donchian_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: Weekly data for pivot points (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, R1 = 2*PP-L, S1 = 2*PP-H, etc.
    # Using previous week's values (already completed)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Weekly pivot calculation
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = pp + 3 * (weekly_high - weekly_low)
    s4 = pp - 3 * (weekly_high - weekly_low)
    
    # Align weekly pivots to 6h timeframe (shifted by 1 for completed weeks only)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # === 6h Indicators: Donchian(20) for breakout detection ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: ATR(14) for stoploss and volatility filter ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume confirmation ===
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
    
    warmup = 100  # Warmup for indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Price ---
        price = close[i]
        
        # --- Weekly Pivot Bias: Determine market bias from weekly PP ---
        # Bullish bias: price above weekly PP
        # Bearish bias: price below weekly PP
        bullish_bias = price > pp_aligned[i]
        bearish_bias = price < pp_aligned[i]
        
        # --- Distance to weekly pivot levels (normalized by ATR) ---
        dist_to_r3 = (price - r3_aligned[i]) / atr_14[i] if atr_14[i] > 0 else 0
        dist_to_s3 = (s3_aligned[i] - price) / atr_14[i] if atr_14[i] > 0 else 0
        dist_to_r4 = (price - r4_aligned[i]) / atr_14[i] if atr_14[i] > 0 else 0
        dist_to_s4 = (s4_aligned[i] - price) / atr_14[i] if atr_14[i] > 0 else 0
        
        # --- Breakout Detection: Donchian(20) breakout with volume confirmation ---
        breakout_up = (price > highest_high[i]) and (vol_ratio[i] > 1.8)
        breakout_down = (price < lowest_low[i]) and (vol_ratio[i] > 1.8)
        
        # --- Mean Reversion Signals at Extreme Weekly Levels ---
        # Fade at R3/S3 with volume exhaustion (volume < average)
        fade_r3 = (dist_to_r3 > 0.5) and (dist_to_r3 < 2.0) and (vol_ratio[i] < 0.7)
        fade_s3 = (dist_to_s3 > 0.5) and (dist_to_s3 < 2.0) and (vol_ratio[i] < 0.7)
        
        # Strong fade at R4/S4 (more extreme levels)
        fade_r4 = (dist_to_r4 > 0.0) and (vol_ratio[i] < 0.6)
        fade_s4 = (dist_to_s4 > 0.0) and (vol_ratio[i] < 0.6)
        
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
                # Exit on mean reversion signal at extreme levels
                if fade_r4 or (fade_r3 and bearish_bias):
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
                # Exit on mean reversion signal at extreme levels
                if fade_s4 or (fade_s3 and bullish_bias):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Breakout entries: Trade in direction of weekly bias
        if breakout_up and bullish_bias:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_down and bearish_bias:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        # Mean reversion entries: Fade extreme weekly levels
        elif fade_s4:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif fade_r4:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        elif fade_s3 and bullish_bias:  # Fade S3 in bullish bias = long
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif fade_r3 and bearish_bias:  # Fade R3 in bearish bias = short
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals