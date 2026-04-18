#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h EMA trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout zones.
# 12h EMA ensures we trade in direction of higher timeframe trend.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above resistance with uptrend) and bear markets (breakdowns below support with downtrend).
name = "4h_Camarilla_12hEMA_Volume_Filter"
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
    
    # Get 12h data for EMA trend (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA (34-period)
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get 1d data for Camarilla pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels using previous day's data to avoid look-ahead
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # Set first value to NaN since there's no previous day
    high_prev[0] = np.nan
    low_prev[0] = np.nan
    close_prev[0] = np.nan
    
    # Calculate pivot and ranges
    pivot = (high_prev + low_prev + close_prev) / 3
    range_val = high_prev - low_prev
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2)
    r4 = pivot + (range_val * 1.1)
    s3 = pivot - (range_val * 1.1 / 2)
    s4 = pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
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
            # Long: price breaks above R4 AND 12h EMA uptrend AND volume confirmation
            long_breakout = close[i] > r4_aligned[i]
            ema_uptrend = close[i] > ema_12h_aligned[i]
            if vol_confirm and long_breakout and ema_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND 12h EMA downtrend AND volume confirmation
            elif vol_confirm and close[i] < s3_aligned[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below R3 OR 12h EMA turns downtrend
            exit_condition = close[i] < r3_aligned[i] or close[i] < ema_12h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S3 OR 12h EMA turns uptrend
            exit_condition = close[i] > s3_aligned[i] or close[i] > ema_12h_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals