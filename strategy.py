#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily high/low/close for pivot calculation
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily pivot point: (H + L + C) / 3
    daily_pivot = (high_daily + low_daily + close_daily) / 3.0
    
    # Calculate daily range and support/resistance levels
    daily_range = high_daily - low_daily
    r1 = 2 * daily_pivot - low_daily
    s1 = 2 * daily_pivot - high_daily
    r2 = daily_pivot + daily_range
    s2 = daily_pivot - daily_range
    
    # Calculate ATR(14) from daily data for volatility filter
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily pivot levels and ATR to 12h timeframe
    daily_pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    atr_14_aligned = align_htf_to_ltf(prices, df_daily, atr_14)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(daily_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price crosses above daily R1 with volume confirmation
            if (close[i] > r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i] and
                atr_14_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses below daily S1 with volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i] and
                  atr_14_aligned[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to daily pivot level
            if position == 1:
                if close[i] < daily_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > daily_pivot_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_DailyPivot_R1S1_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0