#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter and Camarilla pivot
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h data for Camarilla pivot (previous 4h bar)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    pivot = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_4h = prev_high_4h - prev_low_4h
    r1 = pivot + (range_4h * 1.1 / 12)
    s1 = pivot - (range_4h * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Volume filter: 1h volume > 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: Price breaks above R1, price above 4h EMA50, volume above average, in session
            long_cond = (close[i] > r1_aligned[i] and 
                        close[i] > ema50_4h_aligned[i] and
                        volume[i] > vol_ma20[i] and
                        in_session)
            
            # Short: Price breaks below S1, price below 4h EMA50, volume above average, in session
            short_cond = (close[i] < s1_aligned[i] and 
                         close[i] < ema50_4h_aligned[i] and
                         volume[i] > vol_ma20[i] and
                         in_session)
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price closes below S1 OR price crosses below 4h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price closes above R1 OR price crosses above 4h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter, volume confirmation, and session filter (08-20 UTC).
# Uses 4h for signal direction (trend and pivot levels), 1h only for entry timing.
# Session filter reduces noise trades outside active hours. Target: 15-37 trades/year to avoid fee drag.
# Discrete sizing (0.20) minimizes churn. Works in bull via breakout continuation, in bear via mean reversion at S1/R1.