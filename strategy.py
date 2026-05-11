#!/usr/bin/env python3
"""
1d_WeeklyPivot_Breakout_Trend
Hypothesis: Weekly pivot points (P, R1, S1) act as strong support/resistance. 
Price breaking above R1 or below S1 on daily close with 1w trend alignment 
signals momentum continuation. Uses 1d timeframe with 1w pivot and trend filter.
Designed for low trade frequency (<25/year) to avoid fee drag in bull/bear markets.
"""

name = "1d_WeeklyPivot_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for pivot and trend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- Weekly Pivot Points (using previous week's OHLC) ---
    # Calculate from previous week's OHLC
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    # Pivot point calculation
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    
    # Align weekly levels to 1d
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    
    # --- 1w Trend: EMA21 on weekly close ---
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # --- 1d Volume Average for confirmation ---
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for EMA21 and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema21_1w_aligned[i]) or np.isnan(r1_1d[i]) or 
            np.isnan(s1_1d[i]) or np.isnan(vol_avg_1d[i])):
            if position != 0:
                # Simple stoploss: 2.5x ATR from entry
                atr_est = np.abs(high_1d[i] - low_1d[i])  # rough 1d ATR estimate
                if position == 1 and close_1d[i] <= entry_price - 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price + 2.5 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 1d average
        vol_confirm = volume_1d[i] > 1.5 * vol_avg_1d[i]
        
        if position == 0:
            # Look for entries: breakout with trend and volume
            if close_1d[i] > r1_1d[i] and close_1d[i] > ema21_1w_aligned[i] and vol_confirm:
                # Long breakout above R1 with uptrend and volume
                signals[i] = 0.25
                position = 1
                entry_price = close_1d[i]
            elif close_1d[i] < s1_1d[i] and close_1d[i] < ema21_1w_aligned[i] and vol_confirm:
                # Short breakdown below S1 with downtrend and volume
                signals[i] = -0.25
                position = -1
                entry_price = close_1d[i]
        else:
            # Manage existing position: exit on opposite signal or trend change
            if position == 1:
                # Long exit: price below S1 or trend turns down
                if close_1d[i] < s1_1d[i] or close_1d[i] < ema21_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price above R1 or trend turns up
                if close_1d[i] > r1_1d[i] or close_1d[i] > ema21_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals