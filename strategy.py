#!/usr/bin/env python3
# 4h_RVI_Divergence_1dTrend_VolumeFilter
# Hypothesis: Use Relative Vigor Index (RVI) divergence with daily EMA trend filter and volume spike.
# Long when RVI makes higher low while price makes lower low (bullish divergence) with price > daily EMA and volume > 2x MA.
# Short when RVI makes lower high while price makes higher high (bearish divergence) with price < daily EMA and volume > 2x MA.
# Exit when RVI crosses above/below its signal line.
# Designed to capture reversals in both bull and bear markets by filtering with daily trend.
# Targets 20-40 trades/year to minimize fee drag.

name = "4h_RVI_Divergence_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Relative Vigor Index (RVI) calculation
    # Numerator: (close - open) + 2*(close_prev - open_prev) + 2*(close_prev2 - open_prev2) + (close_prev3 - open_prev3)
    # Denominator: (high - low) + 2*(high_prev - low_prev) + 2*(high_prev2 - low_prev2) + (high_prev3 - low_prev3)
    open_ = prices['open'].values
    
    num = (close - open_) + 2 * np.roll(close - open_, 1) + 2 * np.roll(close - open_, 2) + np.roll(close - open_, 3)
    den = (high - low) + 2 * np.roll(high - low, 1) + 2 * np.roll(high - low, 2) + np.roll(high - low, 3)
    
    # Handle first 3 values
    num[:3] = np.nan
    den[:3] = np.nan
    
    # Avoid division by zero
    rvi_raw = np.where(den != 0, num / den, np.nan)
    
    # Signal line: exponential average of RVI
    rvi = pd.Series(rvi_raw).ewm(span=10, adjust=False, min_periods=10).mean().values
    rvi_signal = pd.Series(rvi).ewm(span=4, adjust=False, min_periods=4).mean().values
    
    # Daily EMA for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(rvi[i]) or np.isnan(rvi_signal[i]) or 
            np.isnan(daily_ema_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish divergence: RVI makes higher low while price makes lower low
            bullish_div = (rvi[i] > rvi[i-5]) and (close[i] < close[i-5]) and (rvi[i-5] > rvi[i-10]) and (close[i-5] < close[i-10])
            # Bearish divergence: RVI makes lower high while price makes higher high
            bearish_div = (rvi[i] < rvi[i-5]) and (close[i] > close[i-5]) and (rvi[i-5] < rvi[i-10]) and (close[i-5] > close[i-10])
            
            # LONG: Bullish divergence with price > daily EMA and volume > 2x MA
            if bullish_div and close[i] > daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish divergence with price < daily EMA and volume > 2x MA
            elif bearish_div and close[i] < daily_ema_aligned[i] and volume[i] > vol_ma[i] * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RVI crosses below signal line
            if rvi[i] < rvi_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RVI crosses above signal line
            if rvi[i] > rvi_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals