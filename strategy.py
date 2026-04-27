#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with 1-day EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0, Bear Power < 0, price > 1-day EMA50, and volume > 1.5x average.
# Short when Bull Power < 0, Bear Power > 0, price < 1-day EMA50, and volume > 1.5x average.
# Exit when Bull Power and Bear Power converge (both cross zero) or trend reverses.
# Target: 12-37 trades/year (~50-150 total over 4 years) to avoid fee drag.
# Works in bull/bear via trend-aligned momentum signals.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (standard period)
    ema_period = 13
    ema = np.full(n, np.nan)
    if n >= ema_period:
        ema[ema_period - 1] = np.mean(close[:ema_period])
        for i in range(ema_period, n):
            ema[i] = (close[i] * (2 / (ema_period + 1)) + 
                      ema[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Elder Ray components
    bull_power = high - ema  # Bull Power = High - EMA
    bear_power = low - ema   # Bear Power = Low - EMA
    
    # Calculate 1-day EMA50 for trend filter
    ema_period_1d = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period_1d:
        ema_1d[ema_period_1d - 1] = np.mean(close_1d[:ema_period_1d])
        for i in range(ema_period_1d, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period_1d + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period_1d + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1-day indicators to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need EMA13, EMA50 1d, and volume MA20
    start_idx = max(ema_period - 1, ema_period_1d - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > 1d EMA50, volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, price < 1d EMA50, volume confirmation
            elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power <= 0 or Bear Power >= 0 (convergence) or trend reversal
            if (bull_power[i] <= 0 or bear_power[i] >= 0 or 
                price < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bull Power >= 0 or Bear Power <= 0 (convergence) or trend reversal
            if (bull_power[i] >= 0 or bear_power[i] <= 0 or 
                price > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0