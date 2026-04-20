#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # === Weekly High/Low for Context ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Daily data (primary)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Calculation (based on previous day) ===
    # Shift daily data by 1 to use previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_prev = prev_high - prev_low
    
    # Camarilla levels based on previous day
    R1 = prev_close + (range_prev * 1.1 / 12)
    S1 = prev_close - (range_prev * 1.1 / 12)
    R2 = prev_close + (range_prev * 1.1 / 6)
    S2 = prev_close - (range_prev * 1.1 / 6)
    
    # === Weekly Trend Filter (based on weekly close) ===
    # Use weekly close to determine trend direction
    weekly_close_series = pd.Series(close_1w)
    weekly_ema = weekly_close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Volume Confirmation (daily) ===
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(pivot[i]) or np.isnan(R1[i]) or np.isnan(S1[i]) or
            np.isnan(R2[i]) or np.isnan(S2[i]) or
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly uptrend
            if close[i] > R1[i] and vol_ratio[i] > 2.0 and close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly downtrend
            elif close[i] < S1[i] and vol_ratio[i] > 2.0 and close[i] < weekly_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price falls below S1 or weekly trend turns down
            if close[i] < S1[i] or close[i] < weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises above R1 or weekly trend turns up
            if close[i] > R1[i] or close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals