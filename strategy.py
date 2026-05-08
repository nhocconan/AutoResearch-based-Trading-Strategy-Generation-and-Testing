#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Keltner_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d ATR for Keltner channel (10-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # === 1d EMA20 for middle line ===
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # === Keltner bands (1.5 * ATR) ===
    upper_keltner = ema20_1d_aligned + 1.5 * atr10_1d_aligned
    lower_keltner = ema20_1d_aligned - 1.5 * atr10_1d_aligned
    
    # === 1w EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for volatility and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper Keltner + uptrend + volume
            long_cond = (close[i] > upper_keltner[i] and 
                        close[i] > ema50_1w_aligned[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: break below lower Keltner + downtrend + volume
            short_cond = (close[i] < lower_keltner[i] and 
                         close[i] < ema50_1w_aligned[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below EMA20 (middle line) or opposite signal
            exit_cond = close[i] < ema20_1d_aligned[i]
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above EMA20 (middle line) or opposite signal
            exit_cond = close[i] > ema20_1d_aligned[i]
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Keltner breakout with 1w trend filter and volume confirmation.
# Enters long when price breaks above upper Keltner band (EMA20 + 1.5*ATR) in uptrend
# (price > 1w EMA50) with volume confirmation. Enters short on breakdown below lower
# Keltner band in downtrend with volume. Exits when price crosses back below/above
# the middle line (EMA20). Uses 1d for Keltner calculation, 1w for trend filter.
# Designed to capture trend moves in both bull and bear markets while avoiding
# false breakouts in ranging conditions. Targets 50-150 trades over 4 years.