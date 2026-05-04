#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation
# Uses Elder Ray from 6h to measure bull/bear power relative to EMA13, 12h EMA50 for trend filter (proven from top performers),
# and volume spike for confirmation. Designed for 12-30 trades/year to minimize fee drag.
# Works in bull markets via strong bull power and in bear markets via strong bear power.
# The 12h EMA50 provides a smooth trend filter that avoids whipsaw in ranging markets.

name = "6h_ElderRay_12hEMA50_VolumeSpike_TrendFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter from prior completed 12h bar
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_shifted = np.roll(ema50_12h, 1)
    ema50_12h_shifted[0] = np.nan
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h_shifted)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA of volume on 6h timeframe
        if i >= 20:
            vol_ema_20 = np.mean(volume[i-19:i+1])  # Simple mean for volume spike
            vol_ema_20_prev = np.mean(volume[i-39:i-19]) if i >= 40 else vol_ema_20
            volume_ratio = vol_ema_20 / vol_ema_20_prev if vol_ema_20_prev > 0 else 1.0
        else:
            volume_ratio = 1.0
        
        if position == 0:
            # Long conditions: Bull Power > 0 (strong bullish momentum) AND price above 12h EMA50 AND volume spike
            if bull_power[i] > 0 and close[i] > ema50_12h_aligned[i] and volume_ratio > 1.5:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 (strong bearish momentum) AND price below 12h EMA50 AND volume spike
            elif bear_power[i] < 0 and close[i] < ema50_12h_aligned[i] and volume_ratio > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative OR price closes below 12h EMA50
            if bull_power[i] <= 0 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive OR price closes above 12h EMA50
            if bear_power[i] >= 0 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals