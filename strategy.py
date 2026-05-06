#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Exponential Moving Average (EMA) crossover with 1-week trend filter and volume confirmation
# Uses 1-day EMA(21) and EMA(55) crossover for entry signals
# Requires 1-week EMA(200) to confirm primary trend direction (only trade in direction of weekly trend)
# Volume confirmation (>1.5x 20-day average) ensures institutional participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: trades only with the higher timeframe trend to avoid counter-trend whipsaws

name = "1d_EMACross_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 55 or len(df_1w) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMAs on 1d timeframe
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False).values
    ema_55 = pd.Series(close_1d).ewm(span=55, adjust=False).values
    
    # Calculate 1-week EMA(200) trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    ema_21_aligned = align_htf_to_ltf(prices, df_1d, ema_21)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_21_aligned[i]) or np.isnan(ema_55_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: EMA(21) crosses above EMA(55) AND price above weekly EMA(200) AND volume confirmation
            if (ema_21_aligned[i] > ema_55_aligned[i] and ema_21_aligned[i-1] <= ema_55_aligned[i-1] and
                close[i] > ema_200_1w_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: EMA(21) crosses below EMA(55) AND price below weekly EMA(200) AND volume confirmation
            elif (ema_21_aligned[i] < ema_55_aligned[i] and ema_21_aligned[i-1] >= ema_55_aligned[i-1] and
                  close[i] < ema_200_1w_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: EMA(21) crosses below EMA(55)
            if ema_21_aligned[i] < ema_55_aligned[i] and ema_21_aligned[i-1] >= ema_55_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA(21) crosses above EMA(55)
            if ema_21_aligned[i] > ema_55_aligned[i] and ema_21_aligned[i-1] <= ema_55_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals