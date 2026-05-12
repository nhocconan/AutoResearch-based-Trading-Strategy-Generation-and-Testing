#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Price breaking Camarilla R1/S1 levels on 4h with volume confirmation and 1d trend filter captures institutional breakouts in both bull and bear markets. Designed for low trade frequency (~20-40/year) to minimize fee drag.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # === 1d Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Levels from Previous Day (using 1d OHLC) ===
    # Camarilla formula: Range = (H-L), Levels = C +/- (Range * multiplier)
    # We use previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    range_1d = high_1d - low_1d
    # S1 = C - (Range * 1.1/12), R1 = C + (Range * 1.1/12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    r1 = close_1d + (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # === Volume Confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation (above average)
        vol_confirm = volume[i] > vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                vol_confirm and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  vol_confirm and 
                  trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns below R1 or trend changes
            if (close[i] < r1_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns above S1 or trend changes
            if (close[i] > s1_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals