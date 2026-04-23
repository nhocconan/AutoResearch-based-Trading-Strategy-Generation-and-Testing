#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
Long when Bull Power > 0 AND price > 1w EMA50 AND volume > 1.5x 20-period average.
Short when Bear Power < 0 AND price < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when Elder Power reverses sign OR ATR trailing stop (2.5*ATR from extreme).
Uses 1w HTF for trend alignment (captures major cycles) and daily Elder Ray for signal precision.
Target: ~15-25 trades/year on 6h timeframe with discrete sizing 0.25.
Works in bull via trend alignment and in bear via mean reversion at extremes.
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray from daily data ( Bull Power = High - EMA13, Bear Power = Low - EMA13 )
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of daily close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Daily High - EMA13, Bear Power = Daily Low - EMA13
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Power to 6h timeframe (use previous day's values)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6h volume average (20-period) for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for 6h trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 13)  # vol_ma20, ema_50_1w, ema_13_1d
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_50_1w_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price > 1w EMA50 AND volume spike
            if bull_val > 0 and price > ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Bear Power < 0 AND price < 1w EMA50 AND volume spike
            elif bear_val < 0 and price < ema_val and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Elder Power reverses sign
            if position == 1 and bull_val <= 0:
                exit_signal = True
            elif position == -1 and bear_val >= 0:
                exit_signal = True
            
            # ATR-based trailing stop: 2.5 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_BullBearPower_1wEMA50_Trend_VolumeSpike_PowerReverse_EXIT_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0