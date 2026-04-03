#!/usr/bin/env python3
"""
Experiment #071: 6h Elder Ray + 1d Trend Filter
HYPOTHESIS: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
On 6h timeframe, we take long when Bull Power > 0 AND Bear Power < 0 (bullish imbalance) AND 1d EMA50 > EMA200 (uptrend).
We take short when Bear Power > 0 AND Bull Power < 0 (bearish imbalance) AND 1d EMA50 < EMA200 (downtrend).
This combines momentum (Elder Ray) with trend filter (1d EMA cross) to work in both bull and bear markets.
Target: 75-150 trades over 4 years on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_071_6h_elder_ray_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: EMA50 and EMA200 for trend filter ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align to 6h timeframe (shifted by 1 day for completed bars only)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h Indicators: EMA13 for Elder Ray calculation ===
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13_6h
    bear_power = ema13_6h - low
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 200  # Warmup for 1d EMA200
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1d_6h[i]) or np.isnan(ema200_1d_6h[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter from 1d ---
        is_uptrend = ema50_1d_6h[i] > ema200_1d_6h[i]
        is_downtrend = ema50_1d_6h[i] < ema200_1d_6h[i]
        
        # --- Elder Ray Signals ---
        bull_strong = bull_power[i] > 0  # Buying pressure
        bear_strong = bear_power[i] > 0  # Selling pressure
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            if position_side > 0:  # Long position
                # Exit when bullish pressure fails OR trend turns down
                if not bull_strong or is_downtrend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when bearish pressure fails OR trend turns up
                if not bear_strong or is_uptrend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: Bullish imbalance AND uptrend
        if bull_strong and not bear_strong and is_uptrend:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Bearish imbalance AND downtrend
        elif bear_strong and not bull_strong and is_downtrend:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>