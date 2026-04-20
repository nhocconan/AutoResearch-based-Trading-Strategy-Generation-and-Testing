#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1-day EMA13 trend filter and volume spike
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with 1-day EMA13 up and volume spike
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with 1-day EMA13 down and volume spike
# Uses 1-day trend to avoid counter-trend trades, volume spike for confirmation
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1-day data ONCE for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1-day EMA13 for trend filter
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate 6-day Elder Ray components
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA13 of 6h close for Elder Ray
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13_6h
    # Bear Power = EMA13 - Low
    bear_power = ema13_6h - low
    
    # Volume spike: 6h volume > 2.0 x 20-period average
    volume = prices['volume'].values
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price and EMA levels
        price = close[i]
        ema13 = ema13_6h[i]
        bp = bull_power[i]
        bp_prev = bull_power[i-1] if i > 0 else 0
        be = bear_power[i]
        be_prev = bear_power[i-1] if i > 0 else 0
        ema13_1d = ema13_1d_aligned[i]
        ema13_1d_prev = ema13_1d_aligned[i-1] if i > 0 else ema13_1d
        
        # Trend conditions
        ema13_1d_up = ema13_1d > ema13_1d_prev
        ema13_1d_down = ema13_1d < ema13_1d_prev
        
        # Elder Ray conditions
        bull_rising = bp > bp_prev
        bull_positive = bp > 0
        bear_falling = be < be_prev
        bear_positive = be > 0  # Bear power positive means bearish pressure
        
        if position == 0:
            # Long: Bull Power positive and rising, Bear Power negative, 1-day EMA13 up, volume spike
            if bull_positive and bull_rising and (be < 0) and ema13_1d_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power positive and falling, Bull Power negative, 1-day EMA13 down, volume spike
            elif bear_positive and bear_falling and (bp < 0) and ema13_1d_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative or 1-day EMA13 turns down
            if not bull_positive or not ema13_1d_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns negative or 1-day EMA13 turns up
            if not bear_positive or not ema13_1d_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA13Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0