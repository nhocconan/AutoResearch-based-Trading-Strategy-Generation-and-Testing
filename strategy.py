#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (8/21) with 4h Supertrend filter and volume confirmation
# Uses 4h Supertrend for trend direction (reduces whipsaw in choppy markets)
# 1h EMA(8/21) crossover for precise entry timing within the trend
# Volume spike (>1.5x 20-bar average) confirms momentum strength
# Discrete sizing 0.20 to minimize fee drag; target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: Supertrend adapts to regimes, EMA crossover captures swings, volume ensures participation

name = "1h_EMA8_21_4hSupertrend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend (ATR=10, mult=3.0)
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    
    hl2_4h = (high_4h + low_4h) / 2
    upper_4h = hl2_4h + (3.0 * atr_4h)
    lower_4h = hl2_4h - (3.0 * atr_4h)
    
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    direction_4h = np.full_like(close_4h, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(supertrend_4h[i-1]):
            supertrend_4h[i] = lower_4h[i]
            direction_4h[i] = 1
        else:
            if close_4h[i] > supertrend_4h[i-1]:
                supertrend_4h[i] = max(lower_4h[i], supertrend_4h[i-1])
                direction_4h[i] = 1
            else:
                supertrend_4h[i] = min(upper_4h[i], supertrend_4h[i-1])
                direction_4h[i] = -1
    
    # Calculate 1h EMA(8) and EMA(21)
    close_s = pd.Series(close)
    ema8 = close_s.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume spike filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1h timeframe (primary)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(direction_4h_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA8 > EMA21 AND 4h uptrend (direction=1) AND volume spike
            if ema8[i] > ema21[i] and direction_4h_aligned[i] == 1 and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: EMA8 < EMA21 AND 4h downtrend (direction=-1) AND volume spike
            elif ema8[i] < ema21[i] and direction_4h_aligned[i] == -1 and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: EMA8 < EMA21 (trend change)
            if ema8[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: EMA8 > EMA21 (trend change)
            if ema8[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals