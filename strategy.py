# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use daily trend (EMA34) to filter entries based on 4h Camarilla R1/S1 breakouts with volume confirmation.
# Long when price breaks above R1 with volume in daily uptrend, short when breaks below S1 with volume in daily downtrend.
# Exit when price returns to Camarilla Pivot or daily trend reverses.
# Designed for low frequency (20-40 trades/year) by using 1d for trend and 4h for entry/exit.
# Works in both bull and bear markets by following higher timeframe trend.

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
    
    # === 4h data for Camarilla levels ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels for previous 4h bar
    # Typical price = (high + low + close) / 3
    typical_4h = (high_4h + low_4h + close_4h) / 3
    # Pivot = typical price
    pivot_4h = typical_4h
    # Range = high - low
    range_4h = high_4h - low_4h
    # R1 = close + (range * 1.1 / 12)
    r1_4h = close_4h + (range_4h * 1.1 / 12)
    # S1 = close - (range * 1.1 / 12)
    s1_4h = close_4h - (range_4h * 1.1 / 12)
    
    # Align Camarilla levels to 4h (wait for 4h bar to close)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
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
            # LONG: Price breaks above R1 with volume, in daily uptrend
            if close[i] > r1_4h_aligned[i] and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume, in daily downtrend
            elif close[i] < s1_4h_aligned[i] and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to pivot level or trend reverses
            if close[i] <= pivot_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level or trend reverses
            if close[i] >= pivot_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals