#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray with 1d regime and volume confirmation
# Hypothesis: Elder Ray (Bull/Bear Power) identifies institutional buying/selling pressure.
# Use 1d EMA200 for regime: only take Bull Power signals when price > EMA200 (bullish regime),
# only Bear Power signals when price < EMA200 (bearish regime). Volume confirms institutional participation.
# Works in bull via Bull Power signals in uptrend, in bear via Bear Power signals in downtrend.
# Target: 12-37 trades/year to minimize fee drift.
name = "6h_elder_ray_1d_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for regime
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 1d average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 13-period EMA for Elder Ray (standard period)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1d average volume
        vol_confirm = volume[i] > vol_avg_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR regime changes to bearish
            if bull_power[i] <= 0 or close[i] < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR regime changes to bullish
            if bear_power[i] >= 0 or close[i] > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Bull Power > 0 + price above EMA200 (bullish regime) + volume confirmation
            if bull_power[i] > 0 and close[i] > ema200_1d_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power < 0 + price below EMA200 (bearish regime) + volume confirmation
            elif bear_power[i] < 0 and close[i] < ema200_1d_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals