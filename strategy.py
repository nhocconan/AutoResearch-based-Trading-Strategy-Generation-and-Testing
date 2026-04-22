#!/usr/bin/env python3
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
    
    # Load 12h data for trend filter (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Load 1d data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R4/S4 are key breakout levels)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r4 = close_1d + range_ * 1.1 / 2  # Resistance level 4
    s4 = close_1d - range_ * 1.1 / 2  # Support level 4
    
    # Align all levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume spike AND above 12h EMA50 (uptrend)
            if (close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume spike AND below 12h EMA50 (downtrend)
            elif (close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite R1/S1 level (tighter stop)
            if position == 1:
                # Exit long: Price closes below S1 (calculated from previous day)
                if i > 0:
                    s1 = close_1d[i-1] - (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
                    s1_series = pd.Series(np.full_like(close_1d, s1))
                    s1_aligned_exit = align_htf_to_ltf(prices, df_1d, s1_series.values)[i]
                else:
                    s1_aligned_exit = np.nan
                if not np.isnan(s1_aligned_exit) and close[i] < s1_aligned_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above R1 (calculated from previous day)
                if i > 0:
                    r1 = close_1d[i-1] + (high_1d[i-1] - low_1d[i-1]) * 1.1 / 12
                    r1_series = pd.Series(np.full_like(close_1d, r1))
                    r1_aligned_exit = align_htf_to_ltf(prices, df_1d, r1_series.values)[i]
                else:
                    r1_aligned_exit = np.nan
                if not np.isnan(r1_aligned_exit) and close[i] > r1_aligned_exit:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Camarilla_R4_S4_Breakout_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0