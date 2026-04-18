#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 12h volume filter and 1d EMA34 trend filter.
# Camarilla levels provide high-probability reversal/breakout zones based on prior day's price action.
# Volume filter ensures breakouts have conviction.
# EMA34 filter on 1d ensures we trade in the direction of higher timeframe trend.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 6h timeframe.
# Works in bull markets (breakouts above R3/R4 in uptrend) and bear markets (breakouts below S3/S4 in downtrend).
name = "6h_Camarilla_R3S4_12hVol_1dEMA34Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h volume average for confirmation (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # Using previous day's data to avoid look-ahead
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    h_1d_prev = np.roll(h_1d, 1)
    l_1d_prev = np.roll(l_1d, 1)
    c_1d_prev = np.roll(c_1d, 1)
    # Set first value to NaN since there's no previous day
    h_1d_prev[0] = np.nan
    l_1d_prev[0] = np.nan
    c_1d_prev[0] = np.nan
    
    camarilla_width = (h_1d_prev - l_1d_prev) * 1.1
    r3 = c_1d_prev + camarilla_width / 4
    r4 = c_1d_prev + camarilla_width / 2
    s3 = c_1d_prev - camarilla_width / 4
    s4 = c_1d_prev - camarilla_width / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(c_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume above average
        vol_confirm = volume[i] > vol_ma_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R4 AND volume confirmation AND uptrend (price > EMA34)
            long_breakout = close[i] > r4_aligned[i]
            uptrend = close[i] > ema_34_1d_aligned[i]
            if vol_confirm and long_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND volume confirmation AND downtrend (price < EMA34)
            elif vol_confirm and close[i] < s3_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below R3 OR trend turns down
            exit_condition = close[i] < r3_aligned[i] or close[i] < ema_34_1d_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above S3 OR trend turns up
            exit_condition = close[i] > s3_aligned[i] or close[i] > ema_34_1d_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals