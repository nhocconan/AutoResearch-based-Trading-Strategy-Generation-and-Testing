#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 3-bar EMA crossover with 1w trend filter and volume confirmation
# Uses fast/slow EMA crossover on 1d timeframe for trend direction
# Requires 1w EMA(50) > EMA(200) for long, < for short to filter counter-trend
# Volume confirmation (>1.5x 20-bar average) ensures participation
# Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year)
# Works in both bull/bear: follows major trends only, avoids chop

name = "1d_EMACrossover_1wTrend_Filter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMAs on 1d timeframe
    ema_fast = pd.Series(close_1d).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate EMAs on 1w timeframe for trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR(14) for 1d timeframe (for volatility filter)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume confirmation filter (>1.5x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe (primary)
    ema_fast_aligned = align_htf_to_ltf(prices, df_1d, ema_fast)
    ema_slow_aligned = align_htf_to_ltf(prices, df_1d, ema_slow)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema_fast_aligned[i]) or np.isnan(ema_slow_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: fast EMA > slow EMA AND 1w trend bullish (EMA50 > EMA200) AND volume confirmation
            if (ema_fast_aligned[i] > ema_slow_aligned[i] and 
                ema_50_aligned[i] > ema_200_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: fast EMA < slow EMA AND 1w trend bearish (EMA50 < EMA200) AND volume confirmation
            elif (ema_fast_aligned[i] < ema_slow_aligned[i] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: fast EMA < slow EMA (trend change)
            if ema_fast_aligned[i] < ema_slow_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: fast EMA > slow EMA (trend change)
            if ema_fast_aligned[i] > ema_slow_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals