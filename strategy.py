#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level S3/R3 breakout with 1d EMA34 trend filter and volume confirmation.
# Camarilla pivot levels provide precise intraday support/resistance based on prior day's range.
# S3 and R3 levels act as strong reversal/breakout zones. EMA34 on 1d filters for trend alignment.
# Volume confirmation adds conviction. Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in bull markets (breakouts above R3 with uptrend) and bear markets (breakouts below S3 with downtrend).
name = "4h_Camarilla_S3R3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels using previous day's data to avoid look-ahead
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Pivot point and range
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels (S3, R3)
    s3 = close_prev - range_prev * 1.1 / 2
    r3 = close_prev + range_prev * 1.1 / 2
    
    # Calculate 34-period EMA on daily close for trend filter
    close_series = pd.Series(df_1d['close'])
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily Camarilla levels and EMA to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND EMA34 uptrend (price > EMA) AND volume confirmation
            long_breakout = close[i] > r3_aligned[i]
            uptrend = close[i] > ema_34_aligned[i]
            if vol_confirm and uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND EMA34 downtrend (price < EMA) AND volume confirmation
            elif vol_confirm and (close[i] < ema_34_aligned[i]) and (close[i] < s3_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below R3 OR trend reverses (price < EMA)
            exit_condition = close[i] < r3_aligned[i] or close[i] < ema_34_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above S3 OR trend reverses (price > EMA)
            exit_condition = close[i] > s3_aligned[i] or close[i] > ema_34_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals