#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Elders Ray (Bull/Bear Power) with 12h trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0 (bullish momentum), 12h EMA50 rising (uptrend), and volume spike.
# Short when Bear Power < 0, Bull Power < 0 (bearish momentum), 12h EMA50 falling (downtrend), and volume spike.
# Uses Elder Ray for momentum strength, 12h EMA50 for trend filter, volume to confirm participation.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via long positions in uptrend and in bear markets via short positions in downtrend.
name = "6s_ElderRay_12hTrend_Volume"
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
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_rising = ema_50_12h > ema_50_12h_prev
    ema_falling = ema_50_12h < ema_50_12h_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA13 and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, EMA rising, volume spike
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and ema_rising_aligned[i] and volume_spike[i]
            # Short: Bear Power < 0, Bull Power < 0, EMA falling, volume spike
            short_condition = (bear_power[i] < 0) and (bull_power[i] < 0) and ema_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or EMA turns flat/falling
            if bull_power[i] <= 0 or not ema_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or EMA turns flat/rising
            if bear_power[i] >= 0 or not ema_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals