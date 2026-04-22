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
    
    # Load 1d data for pivot points and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
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
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34 = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume spike AND above 1d EMA34 (uptrend)
            if (close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume spike AND below 1d EMA34 (downtrend)
            elif (close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                  close[i] < ema_34_aligned[i]):
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

name = "12H_Camarilla_R4_S4_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0