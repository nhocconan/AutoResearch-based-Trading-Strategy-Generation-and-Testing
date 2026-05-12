#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakout on 4h with 1d EMA34 trend filter and volume confirmation.
# Camarilla levels provide high-probability support/resistance zones. The 1d EMA34 filter ensures
# alignment with daily trend, reducing counter-trend trades. Volume confirmation ensures breakouts
# have conviction. Designed to work in both bull and bear markets by following the trend defined
# by higher timeframe. Uses discrete position sizing (0.25) to minimize fee churn.

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
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Levels from Previous 1d Bar ===
    # Calculate Camarilla R1, S1 from previous 1d bar
    # We'll use the previous 1d bar's high, low, close to calculate levels for current 4h bar
    # The Camarilla levels are based on the previous day's range
    # We need to get the previous 1d bar's high, low, close for each 4h bar
    # Instead, we calculate the Camarilla levels for each 1d bar and then align to 4h
    
    # Calculate typical price for 1d bars
    typical_price_1d = (high_1d := df_1d['high'].values) + (low_1d := df_1d['low'].values) + close_1d
    typical_price_1d = typical_price_1d / 3.0
    
    # Calculate Camarilla R1 and S1 for each 1d bar
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === Volume Confirmation (20-period average on 4h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction from 1d EMA
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume and daily uptrend
            if (close[i] > r1_1d_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume and daily downtrend
            elif (close[i] < s1_1d_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below S1 or daily trend changes
            if (close[i] < s1_1d_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above R1 or daily trend changes
            if (close[i] > r1_1d_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals