# 12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
# Hypothesis: Camarilla pivot breakout on 12h chart with weekly trend filter and volume confirmation.
# Uses precise Camarilla levels (R1/S1) for high-probability entries with strict conditions.
# Weekly trend filter (EMA20) ensures alignment with higher timeframe momentum.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (<30/year) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes via trend filter).

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate Camarilla levels
    # R1 = Close + (High - Low) * 1.12
    # S1 = Close - (High - Low) * 1.12
    camarilla_range = prev_high - prev_low
    r1_level = prev_close + camarilla_range * 1.12
    s1_level = prev_close - camarilla_range * 1.12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    # Initialize first 20 values
    for i in range(20):
        vol_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_20_1w_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: Close > R1 AND price > weekly EMA20 (uptrend) AND volume > 2x average
            if close[i] > r1 and close[i] > ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < S1 AND price < weekly EMA20 (downtrend) AND volume > 2x average
            elif close[i] < s1 and close[i] < ema_1w and vol > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < S1 OR trend reverses (price < weekly EMA20)
            if close[i] < s1 or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > R1 OR trend reverses (price > weekly EMA20)
            if close[i] > r1 or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals