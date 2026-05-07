#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_BearPower_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    weekly_up = close > ema_50_1w_aligned
    weekly_down = close < ema_50_1w_aligned
    
    # Daily data for Elder Ray (13-period EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_aligned
    bear_power = low - ema13_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~18 hours (3*6h) to reduce trade frequency
    
    start_idx = max(13, 1)  # Ensure enough data for EMA13
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Weekly uptrend AND Bull Power > 0 (bullish momentum)
            if weekly_up[i] and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Weekly downtrend AND Bear Power < 0 (bearish momentum)
            elif weekly_down[i] and bear_power[i] < 0:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Weekly trend turns down OR Bull Power turns negative
            if not weekly_up[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Weekly trend turns up OR Bear Power turns positive
            if not weekly_down[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull Power/Bear Power) with weekly trend filter captures momentum in both bull and bear markets.
# Bull Power > 0 indicates bullish momentum (high > EMA13), Bear Power < 0 indicates bearish momentum (low < EMA13).
# Weekly trend filter ensures we only take longs in weekly uptrends and shorts in weekly downtrends.
# This avoids counter-trend whipsaws. The 13-period EMA on daily data provides smooth momentum measurement.
# Cooldown of 3 bars (18 hours) limits trades to ~20-50 per year. Position size 0.25 manages risk.
# Works in bull markets (captures uptrends) and bear markets (captures downtrends).