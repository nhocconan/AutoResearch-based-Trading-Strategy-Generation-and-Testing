#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 12-hour trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13(Close), Bear Power = EMA13(Close) - Low
# Long when: Bull Power > 0 AND EMA13(12h) rising AND volume > 1.5 * EMA20(volume)
# Short when: Bear Power > 0 AND EMA13(12h) falling AND volume > 1.5 * EMA20(volume)
# Exit when Bull/Bear Power crosses back below zero.
# Designed for low trade frequency (target: 12-37/year on 6h) to minimize fee drag.
# Works in bull markets via Bull Power and in bear markets via Bear Power.
name = "6h_ElderRay_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Elder Ray: EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # EMA13 for exit (same as Elder Ray EMA)
    
    # Load 12h data for EMA13 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA13 on 12h close
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Rising if current > previous, falling if current < previous
    ema13_rising = np.zeros_like(ema13_12h, dtype=bool)
    ema13_falling = np.zeros_like(ema13_12h, dtype=bool)
    ema13_rising[1:] = ema13_12h[1:] > ema13_12h[:-1]
    ema13_falling[1:] = ema13_12h[1:] < ema13_12h[:-1]
    
    ema13_rising_aligned = align_htf_to_ltf(prices, df_12h, ema13_rising)
    ema13_falling_aligned = align_htf_to_ltf(prices, df_12h, ema13_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema13[i]) or 
            np.isnan(ema13_rising_aligned[i]) or np.isnan(ema13_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND EMA13(12h) rising AND volume spike
            long_condition = (bull_power[i] > 0) and ema13_rising_aligned[i] and volume_spike[i]
            # Short: Bear Power > 0 AND EMA13(12h) falling AND volume spike
            short_condition = (bear_power[i] > 0) and ema13_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power <= 0
            if bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals