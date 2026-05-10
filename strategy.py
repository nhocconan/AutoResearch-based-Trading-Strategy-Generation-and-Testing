#!/usr/bin/env python3
# 12h_Williams_Alligator_ElderRay_Signal
# Hypothesis: Combine Williams Alligator trend direction with Elder Ray power for high-conviction trades.
# Use 1-day trend filter to avoid counter-trend trades. Enter on Alligator alignment with bullish/bearish Elder Ray.
# Williams Alligator: Jaw (13-bar SMMA, 8 offset), Teeth (8-bar SMMA, 5 offset), Lips (5-bar SMMA, 3 offset).
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when: Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0 AND 1-day Uptrend.
# Short when: Jaw > Teeth > Lips (bearish alignment) AND Bear Power > 0 AND 1-day Downtrend.
# Position size: 0.25. Max 1 trade every 5 bars to reduce churn.
# Works in bull/bear by following 1-day trend and using Alligator for entry timing.

name = "12h_Williams_Alligator_ElderRay_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA)"""
    if length < 1:
        return np.full_like(source, np.nan, dtype=float)
    result = np.full_like(source, np.nan, dtype=float)
    if len(source) == 0:
        return result
    result[0] = source[0]
    for i in range(1, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Williams Alligator components
    close_s = pd.Series(close)
    jaw = smma(close_s.values, 13)  # 13-period SMMA
    teeth = smma(close_s.values, 8)  # 8-period SMMA
    lips = smma(close_s.values, 5)   # 5-period SMMA
    
    # Apply offsets: Jaw +8, Teeth +5, Lips +3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set rolled values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Elder Ray components
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # 1-day trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_last_trade += 1
        
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0 and bars_since_last_trade >= 5:
            # Enter long: bullish alignment + bullish power + daily uptrend
            if bullish_alignment and bull_power[i] > 0 and daily_up:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Enter short: bearish alignment + bearish power + daily downtrend
            elif bearish_alignment and bear_power[i] > 0 and daily_down:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        
        elif position == 1:
            # Exit conditions: alignment breaks or power fades
            if not bullish_alignment or bull_power[i] <= 0 or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: alignment breaks or power fades
            if not bearish_alignment or bear_power[i] <= 0 or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals