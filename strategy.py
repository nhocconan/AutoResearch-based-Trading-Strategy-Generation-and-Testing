#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    trend_up = close > ema_1d_aligned
    
    # 1d volume filter: volume > 1.5x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.5 * vol_ma20_1d_aligned
    
    # Daily Camarilla levels (based on previous day's high-low-close)
    # We need previous day's data, so we shift the daily data by 1
    close_1d_shift = np.roll(close_1d, 1)
    high_1d_shift = np.roll(high, 1)  # Note: this is a simplification, ideally we'd use daily high/low
    low_1d_shift = np.roll(low, 1)
    
    # For simplicity, we'll use the current day's high-low-close for Camarilla calculation
    # In practice, we should use previous day's, but for now we'll approximate
    # To get proper previous day's data, we need to access the actual daily dataframe
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align the previous day's data to the 4h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = prev_high_aligned - prev_low_aligned
    r1 = prev_close_aligned + 1.1 * camarilla_range / 12
    s1 = prev_close_aligned - 1.1 * camarilla_range / 12
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for EMA and shifted values
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(prev_close_aligned[i]) or np.isnan(prev_high_aligned[i]) or
            np.isnan(prev_low_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close above R1 + daily uptrend + volume filter
            if close[i] > r1[i] and trend_up[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S1 + daily downtrend + volume filter
            elif close[i] < s1[i] and not trend_up[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close below S1 or daily trend down
            if close[i] < s1[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close above R1 or daily trend up
            if close[i] > r1[i] or trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals