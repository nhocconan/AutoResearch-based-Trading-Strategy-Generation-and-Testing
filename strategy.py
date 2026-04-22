#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to 12h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Load daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate daily EMA34 for additional trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily OHLC for Camarilla pivot levels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Previous day's OHLC for today's pivot levels
    high_prev = np.roll(high_d, 1)
    low_prev = np.roll(low_d, 1)
    close_prev = np.roll(close_d, 1)
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    
    # Camarilla levels (S1 and R1)
    s1 = close_prev - (range_val * 1.1 / 12)
    r1 = close_prev + (range_val * 1.1 / 12)
    
    # Align all levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after lookback
        # Skip if data not ready
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price closes above R1 with bullish weekly and daily trend, volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema20_1w_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and
                volume[i] > 2.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price closes below S1 with bearish weekly and daily trend, volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema20_1w_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  volume[i] > 2.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to pivot point
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot
                if close[i] <= pivot_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot
                if close[i] >= pivot_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1wEMA20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0