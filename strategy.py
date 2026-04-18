#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume spike and volatility filter.
# Camarilla pivot levels (R1, S1) from daily chart provide precise entry/exit points.
# Volume spike confirms breakout conviction. Daily ATR filter ensures sufficient volatility.
# Designed for low trade frequency (12-37/year) to minimize fee drag in 12h timeframe.
# Works in bull markets (breakouts above R1) and bear markets (breakouts below S1).
name = "12h_Camarilla_R1_S1_Breakout_Volume_ATRFilter"
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
    
    # Get daily data for Camarilla pivots and ATR (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day's data
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Calculate pivot components using previous day's data to avoid look-ahead
    range_d = high_d - low_d
    r1_d = close_d + range_d * 1.1 / 12
    s1_d = close_d - range_d * 1.1 / 12
    
    # Shift to use previous day's levels (avoid look-ahead)
    r1_d = np.roll(r1_d, 1)
    s1_d = np.roll(s1_d, 1)
    r1_d[0] = np.nan
    s1_d[0] = np.nan
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = high_d[1:] - low_d[1:]
    tr2 = np.abs(high_d[1:] - close_d[:-1])
    tr3 = np.abs(low_d[1:] - close_d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]) and not np.isnan(tr[i]):
                atr[i] = atr[i-1] * (1 - 1/atr_period) + tr[i] * (1/atr_period)
            else:
                atr[i] = np.nan
    
    # ATR multiplier for volatility filter
    atr_mult = 1.5
    atr_threshold = atr * atr_mult
    
    # Align daily R1, S1, and ATR threshold to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_d)
    atr_threshold_12h = align_htf_to_ltf(prices, df_1d, atr_threshold)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(atr_threshold_12h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        # Volatility filter: current ATR threshold must be positive (sufficient volatility)
        vol_filter = not np.isnan(atr_threshold_12h[i]) and atr_threshold_12h[i] > 0
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirmation AND volatility filter
            long_breakout = close[i] > r1_12h[i]
            if vol_confirm and vol_filter and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume confirmation AND volatility filter
            elif vol_confirm and vol_filter and close[i] < s1_12h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below S1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] < s1_12h[i] or (np.isnan(atr_threshold_12h[i]) or atr_threshold_12h[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 OR ATR drops below threshold (volatility collapse)
            exit_condition = close[i] > r1_12h[i] or (np.isnan(atr_threshold_12h[i]) or atr_threshold_12h[i] <= 0)
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals