#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels from daily timeframe. Long when price breaks above R1 with volume confirmation and price above daily EMA34 (uptrend).
# Short when price breaks below S1 with volume confirmation and price below daily EMA34 (downtrend). Exit when price returns to the central pivot (PP).
# Designed for low frequency (12-37 trades/year) to capture momentum in both bull and bear markets by aligning with higher timeframe trend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # === 1d data for Camarilla pivot and trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and levels for previous day (shifted by 1 to avoid look-ahead)
    pp = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    r1 = pp + (high_1d[:-1] - low_1d[:-1]) * 1.1 / 12
    s1 = pp - (high_1d[:-1] - low_1d[:-1]) * 1.1 / 12
    
    # Align to 12h timeframe (these levels are valid for the entire day)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to or below central pivot (PP)
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or above central pivot (PP)
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals