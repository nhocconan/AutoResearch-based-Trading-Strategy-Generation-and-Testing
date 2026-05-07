#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 12-hour trend filter and volume confirmation.
# Long when: Bull Power > 0, Bear Power < 0, 12h EMA50 rising, volume > 1.5x EMA20(volume).
# Short when: Bull Power < 0, Bear Power > 0, 12h EMA50 falling, volume > 1.5x EMA20(volume).
# Exit when Elder Ray signals reverse or volume drops below average.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via strong bullish energy and in bear markets via strong bearish energy.
name = "6h_ElderRay_12hTrend_Volume"
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
    
    # Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # EMA20 for volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_prev = np.roll(ema50_12h, 1)
    ema50_12h_prev[0] = ema50_12h[0]  # avoid NaN at start
    ema50_rising = ema50_12h > ema50_12h_prev
    ema50_falling = ema50_12h < ema50_12h_prev
    
    ema50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema50_rising)
    ema50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema50_falling)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, 12h EMA50 rising, volume spike
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and ema50_rising_aligned[i] and volume_spike
            # Short: Bull Power < 0, Bear Power > 0, 12h EMA50 falling, volume spike
            short_condition = (bull_power[i] < 0) and (bear_power[i] > 0) and ema50_falling_aligned[i] and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or Bear Power >= 0 or no volume spike
            if (bull_power[i] <= 0) or (bear_power[i] >= 0) or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power >= 0 or Bear Power <= 0 or no volume spike
            if (bull_power[i] >= 0) or (bear_power[i] <= 0) or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals