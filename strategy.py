#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index with 1-day trend filter and volume confirmation.
# Long when: Bull Power > 0 AND Bear Power < 0 AND EMA34(1d) rising AND volume > 1.5 * EMA20(volume).
# Short when: Bear Power < 0 AND Bull Power < 0 AND EMA34(1d) falling AND volume > 1.5 * EMA20(volume).
# Exit when Bull Power and Bear Power have same sign (both positive or both negative).
# Elder Ray measures bull/bear power relative to EMA; trend filter ensures alignment with higher timeframe;
# volume confirms conviction. Designed for low trade frequency (target: 15-30/year) to minimize fee drag.
# Works in bull markets via sustained bull power and in bear markets via sustained bear power.
name = "6h_ElderRay_1dEMA34_Volume"
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
    
    # EMA13 for Elder Ray calculation (standard period)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_34_rising_aligned[i]) or 
            np.isnan(ema_34_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 AND EMA34(1d) rising AND volume spike
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Bear Power < 0 AND Bull Power < 0 AND EMA34(1d) falling AND volume spike
            short_condition = (bear_power[i] < 0) and (bull_power[i] < 0) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power and Bear Power have same sign (both >= 0 or both <= 0)
            if (bull_power[i] >= 0 and bear_power[i] >= 0) or (bull_power[i] <= 0 and bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power and Bear Power have same sign (both >= 0 or both <= 0)
            if (bull_power[i] >= 0 and bear_power[i] >= 0) or (bull_power[i] <= 0 and bear_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals