#!/usr/bin/env python3
"""
Hypothesis: Daily price action often respects weekly pivot levels, with mean reversion to the weekly pivot point
during ranging markets and breakouts beyond weekly support/resistance during trending markets.
By combining the weekly pivot point with a daily EMA50 trend filter and volume confirmation,
we aim to capture both mean-reversion and breakout opportunities on the 12-hour timeframe.
The strategy enters long when price crosses above the weekly pivot with volume > 1.8x average
and price above daily EMA50, and short when price crosses below the weekly pivot with volume > 1.8x
average and price below daily EMA50. Exits occur when price returns to the midpoint between pivot
and the prior week's high/low. Designed for 12h timeframe to work in bull (breakouts) and bear
(mean reversion to weekly pivot) regimes with ~15-30 trades per year.
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot and support/resistance levels
    wheek_high = df_1w['high'].values
    wheek_low = df_1w['low'].values
    wheek_close = df_1w['close'].values
    
    pivot = (wheek_high + wheek_low + wheek_close) / 3
    range_ = wheek_high - wheek_low
    
    # Define exit levels: midpoint between pivot and prior week's high/low
    upper_exit = (pivot + wheek_high) / 2
    lower_exit = (pivot + wheek_low) / 2
    
    # Calculate daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    dclose = df_1d['close'].values
    ema_50 = pd.Series(dclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly levels to 12h timeframe (waits for weekly bar to close)
    pivot_12h = align_htf_to_ltf(prices, df_1w, pivot)
    upper_exit_12h = align_htf_to_ltf(prices, df_1w, upper_exit)
    lower_exit_12h = align_htf_to_ltf(prices, df_1w, lower_exit)
    
    # Align daily EMA50 to 12h timeframe
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_12h[i]) or np.isnan(upper_exit_12h[i]) or np.isnan(lower_exit_12h[i]) or
            np.isnan(ema_50_12h[i]) or np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price crosses above weekly pivot with volume spike and above daily EMA50
            if price > pivot_12h[i] and vol > 1.8 * vol_ma and price > ema_50_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly pivot with volume spike and below daily EMA50
            elif price < pivot_12h[i] and vol > 1.8 * vol_ma and price < ema_50_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to upper exit level (midpoint between pivot and prior week high)
            if price < upper_exit_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to lower exit level (midpoint between pivot and prior week low)
            if price > lower_exit_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WeeklyPivot_Volume_EMA50_MidExit"
timeframe = "12h"
leverage = 1.0