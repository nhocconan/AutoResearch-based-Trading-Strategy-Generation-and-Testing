#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1-day Elder Ray index (bull/bear power) with 6-hour MACD histogram crossovers.
# Elder Ray uses 13-period EMA of daily closes to compute bull power (high - EMA) and bear power (low - EMA).
# Bullish when daily bear power is rising (less negative) and 6h MACD histogram crosses above zero.
# Bearish when daily bull power is falling (less positive) and 6h MACD histogram crosses below zero.
# Uses volume confirmation (volume > 1.5x 20-period average) to filter weak breakouts.
# Position size 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Designed to work in both bull and bear markets by using daily power dynamics and momentum confirmation.

name = "6h_1dElderRay_6hMACD_Volume"
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
    
    # Get 1d data for Elder Ray (bull/bear power)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 6h data for MACD
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 35:  # Need enough for MACD (26+9)
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    
    # 1-day EMA(13) for Elder Ray
    ema_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_1d  # Higher = stronger bulls
    bear_power = low_1d - ema_1d   # Lower (more negative) = stronger bears
    
    # Slope of bear power: rising (less negative) = bullish signal
    # Slope of bull power: falling (less positive) = bearish signal
    bear_power_slope = pd.Series(bear_power).diff().values
    bull_power_slope = pd.Series(bull_power).diff().values
    
    bear_power_slope_aligned = align_htf_to_ltf(prices, df_1d, bear_power_slope)
    bull_power_slope_aligned = align_htf_to_ltf(prices, df_1d, bull_power_slope)
    
    # 6-hour MACD(12,26,9)
    ema_fast = pd.Series(close_6h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close_6h).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    macd_hist_aligned = align_htf_to_ltf(prices, df_6h, macd_hist)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bear_power_slope_aligned[i]) or np.isnan(bull_power_slope_aligned[i]) or
            np.isnan(macd_hist_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bear power rising (less negative) AND MACD hist crosses above zero, volume confirmation
            if (bear_power_slope_aligned[i] > 0 and
                macd_hist_aligned[i] > 0 and
                macd_hist_aligned[i-1] <= 0 and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: bull power falling (less positive) AND MACD hist crosses below zero, volume confirmation
            elif (bull_power_slope_aligned[i] < 0 and
                  macd_hist_aligned[i] < 0 and
                  macd_hist_aligned[i-1] >= 0 and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: bear power falling (more negative) OR MACD hist crosses below zero
            if (bear_power_slope_aligned[i] < 0 or 
                (macd_hist_aligned[i] < 0 and macd_hist_aligned[i-1] >= 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power rising (more positive) OR MACD hist crosses above zero
            if (bull_power_slope_aligned[i] > 0 or 
                (macd_hist_aligned[i] > 0 and macd_hist_aligned[i-1] <= 0)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals