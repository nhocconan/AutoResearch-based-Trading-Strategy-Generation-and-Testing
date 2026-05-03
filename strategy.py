#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA(50) trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In strong uptrends, Bull Power expands and stays positive; in downtrends, Bear Power expands and stays negative
# We use 6h for entry timing, 12h EMA(50) for trend alignment, and volume spike for confirmation
# Long: Bull Power > 0 AND Bear Power < 0 (bulls in control) AND price > 12h EMA50 AND volume spike
# Short: Bear Power < 0 AND Bull Power > 0 (bears in control) AND price < 12h EMA50 AND volume spike
# This avoids whipsaws by requiring both bull and bear power to confirm regime
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing trends

name = "6h_ElderRay_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components on 6h: need EMA13 first
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Elder Ray signals with 12h trend filter
        # Long: Bull Power > 0 (bulls in control) AND Bear Power < 0 (no bear pressure) 
        #       AND price above 12h EMA50 AND volume spike
        # Short: Bear Power < 0 (bears in control) AND Bull Power > 0 (no bull pressure)
        #        AND price below 12h EMA50 AND volume spike
        if position == 0:
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_50_12h_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and bull_power[i] > 0 and close[i] < ema_50_12h_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bears take over (Bear Power > 0) OR price breaks below 12h EMA50
            if bear_power[i] > 0 or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bulls take over (Bull Power < 0) OR price breaks above 12h EMA50
            if bull_power[i] < 0 or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals