#!/usr/bin/env python3
name = "6h_WeeklyPivot_Momentum_Breakout_v4"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    
    # Momentum filter: 6h RSI(14) > 50 for long, < 50 for short
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h to reduce trades
    
    start_idx = max(200, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Price relative to weekly pivot levels
        price_above_r2 = close[i] > weekly_r2_aligned[i]
        price_below_s2 = close[i] < weekly_s2_aligned[i]
        price_between_r1_r2 = (close[i] > weekly_r1_aligned[i]) and (close[i] < weekly_r2_aligned[i])
        price_between_s1_s2 = (close[i] > weekly_s2_aligned[i]) and (close[i] < weekly_s1_aligned[i])
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above weekly R2 with bullish momentum and volume
            if price_above_r2 and (rsi[i] > 50) and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below weekly S2 with bearish momentum and volume
            elif price_below_s2 and (rsi[i] < 50) and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below weekly R1 or momentum turns bearish
            if (close[i] < weekly_r1_aligned[i]) or (rsi[i] < 50):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises back above weekly S1 or momentum turns bullish
            if (close[i] > weekly_s1_aligned[i]) or (rsi[i] > 50):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot points act as strong support/resistance levels. 
# Breakout above weekly R2 with bullish momentum (RSI>50) and volume confirmation signals continuation of uptrend.
# Breakdown below weekly S2 with bearish momentum (RSI<50) and volume signals continuation of downtrend.
# Uses weekly timeframe for pivot calculation to capture major market structure.
# RSI(14) filter ensures momentum alignment, reducing false breakouts.
# Volume filter confirms institutional participation.
# Designed to work in both bull and bear markets by capturing directional breaks of key weekly levels.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.