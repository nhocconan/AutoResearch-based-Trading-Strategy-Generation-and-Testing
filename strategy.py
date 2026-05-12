#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 breakouts on 12h with 1d EMA34 trend filter and volume confirmation.
# Uses Camarilla pivot levels for precise entry/exit in ranging markets and EMA trend filter to avoid counter-trend trades.
# Target: 12-37 trades/year (50-150 total over 4 years). Works in bull (breakouts with trend) and bear (mean reversion at extremes via trend filter).

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
    
    # === 1d Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla Levels (using previous 1d OHLC) ===
    # Calculate Camarilla levels from previous day's range
    # R1 = Close + 1.1 * (High - Low) / 12
    # S1 = Close - 1.1 * (High - Low) / 12
    # We need previous day's data, so we'll shift by 1 day
    
    # Get 1d OHLC data
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R1 = C + 1.1*(H-L)/12
    # S1 = C - 1.1*(H-L)/12
    camarilla_r1_1d = close_1d_arr + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1_1d = close_1d_arr - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_12h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_12h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # === Volume Confirmation (20-period average on 12h) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r1_12h_aligned[i]) or 
            np.isnan(camarilla_s1_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction based on 1d EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with volume and uptrend
            if (close[i] > camarilla_r1_12h_aligned[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S1 with volume and downtrend
            elif (close[i] < camarilla_s1_12h_aligned[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to Camarilla S1 or trend changes
            if (close[i] < camarilla_s1_12h_aligned[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to Camarilla R1 or trend changes
            if (close[i] > camarilla_r1_12h_aligned[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals