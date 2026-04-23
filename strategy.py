#!/usr/bin/env python3
"""
Hypothesis: 1h EMA(8)/EMA(21) crossover with 4h Supertrend trend filter and volume confirmation.
Target: 15-37 trades/year per symbol (60-150 total over 4 years). Uses discrete position sizing (0.20) to minimize fee churn.
Uses 4h Supertrend for primary trend direction (works in bull/bear via ATR-based dynamic stop) and 1h EMA crossover for precise timing.
Volume confirmation avoids false breakouts in low-participation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Supertrend for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR calculation
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_4h = (high_4h + low_4h) / 2
    upper_band_4h = hl2_4h + 3.0 * atr_4h
    lower_band_4h = hl2_4h - 3.0 * atr_4h
    
    supertrend_4h = np.full_like(close_4h, np.nan, dtype=float)
    direction_4h = np.ones_like(close_4h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i-1] > upper_band_4h[i-1]:
            direction_4h[i] = -1
        elif close_4h[i-1] < lower_band_4h[i-1]:
            direction_4h[i] = 1
        else:
            direction_4h[i] = direction_4h[i-1]
        
        if direction_4h[i] == 1:
            upper_band_4h[i] = min(upper_band_4h[i], upper_band_4h[i-1])
            lower_band_4h[i] = lower_band_4h[i-1]
        else:
            upper_band_4h[i] = upper_band_4h[i-1]
            lower_band_4h[i] = max(lower_band_4h[i], lower_band_4h[i-1])
        
        if direction_4h[i] == 1:
            supertrend_4h[i] = lower_band_4h[i]
        else:
            supertrend_4h[i] = upper_band_4h[i]
    
    # Align Supertrend direction to 1h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_4h, direction_4h.astype(float))
    
    # Calculate 1h EMA(8) and EMA(21) for entry timing
    ema_8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(21, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(ema_8[i]) or 
            np.isnan(ema_21[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h Supertrend: 1 = uptrend, -1 = downtrend
        trend_up = supertrend_dir_aligned[i] == 1
        trend_down = supertrend_dir_aligned[i] == -1
        
        # EMA crossover signals
        ema_bullish = ema_8[i] > ema_21[i]
        ema_bearish = ema_8[i] < ema_21[i]
        
        # Volume filter: current volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: EMA bullish crossover AND 4h uptrend AND volume confirmation
            if ema_bullish and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: EMA bearish crossover AND 4h downtrend AND volume confirmation
            elif ema_bearish and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: reverse EMA crossover or trend change
            exit_signal = False
            if position == 1:
                # Exit long on EMA bearish crossover or 4h trend turns down
                if ema_bearish or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Exit short on EMA bullish crossover or 4h trend turns up
                if ema_bullish or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_EMA8_21_Crossover_4hSupertrend_Trend_VolumeFilter"
timeframe = "1h"
leverage = 1.0