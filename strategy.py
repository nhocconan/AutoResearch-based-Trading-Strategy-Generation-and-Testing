#!/usr/bin/env python3
# 4H_1D_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS_v2
# Hypothesis: Add volatility filter to avoid whipsaws. Use daily ATR to ensure volatility is above median.
# In low volatility, breakouts often fail. This should reduce false signals and improve win rate.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4H_1D_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and Camarilla levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = pivot + range_ * 1.1 / 4  # R1 = pivot + (range * 1.1 / 4)
    s1 = pivot - range_ * 1.1 / 4  # S1 = pivot - (range * 1.1 / 4)
    
    # Get daily data for EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get daily ATR for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])  # First value is NaN
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    # Calculate median ATR over long period for filter
    atr_median = pd.Series(atr14_1d).rolling(window=50, min_periods=50).median().values
    volatility_filter = atr14_1d > atr_median  # Only trade when volatility is above median
    
    # Align to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 + above daily EMA34 + volume confirmation + volatility filter
            if (close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and 
                volume_confirm[i] and volatility_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below daily EMA34 + volume confirmation + volatility filter
            elif (close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and 
                  volume_confirm[i] and volatility_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below daily EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above daily EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals