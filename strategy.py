#!/usr/bin/env python3
"""
12H_1W_Trend_Retracement_With_Volume
Hypothesis: Strong weekly trend with 12h retracement entries. Uses 1w EMA50 for trend direction, 
12h pullback to EMA21 for entry, and volume confirmation. Weekly trend filter reduces false signals 
in ranging markets. Designed for low turnover (~15-25 trades/year) to minimize fee drag.
Works in both bull and bear by following the dominant weekly trend.
"""

name = "12H_1W_Trend_Retracement_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h Indicators ===
    # EMA21 for retracement entries
    ema21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Volume confirmation: 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Weekly Trend Filter (EMA50) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for weekly EMA and 12h EMA)
    start_idx = 60  # covers EMA21 and weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema21[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from weekly EMA50
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Price position relative to 12h EMA21 (retracement signals)
        price_near_ema21_from_below = close[i] >= ema21[i] * 0.995 and close[i] <= ema21[i]  # Within 0.5% below EMA21
        price_near_ema21_from_above = close[i] <= ema21[i] * 1.005 and close[i] >= ema21[i]  # Within 0.5% above EMA21
        
        if position == 0:
            # Long: Weekly uptrend + price retracing to EMA21 from below + volume
            if weekly_uptrend and price_near_ema21_from_below and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Weekly downtrend + price retracing to EMA21 from above + volume
            elif weekly_downtrend and price_near_ema21_from_above and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: trend reversal or extended move
            if position == 1:
                # Exit: weekly trend turns down OR price moves significantly above EMA21
                if not weekly_uptrend or close[i] > ema21[i] * 1.02:  # 2% above EMA21
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: weekly trend turns up OR price moves significantly below EMA21
                if not weekly_downtrend or close[i] < ema21[i] * 0.98:  # 2% below EMA21
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals