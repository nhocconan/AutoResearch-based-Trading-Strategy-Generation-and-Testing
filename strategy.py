# 12H_Camarilla_R1_S1_1dTrend_Volume_Session
# Hypothesis: 12-hour Camarilla R1/S1 breakout with 1-day EMA trend filter and volume confirmation.
# Uses daily EMA200 to capture long-term trend direction, avoiding counter-trend entries.
# Volume spike (2x 20-period average) confirms breakout momentum.
# Session filter (08-20 UTC) reduces noise during low-volume periods.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by aligning with higher timeframe trend.

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
    
    # Load 1d data for trend and pivot calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1d Camarilla pivots (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    rango = high_1d - low_1d
    r1 = close_1d_prev + (rango * 1.1 / 12)
    s1 = close_1d_prev - (rango * 1.1 / 12)
    
    # Align pivots to 12h timeframe (previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg_20[i])):
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
            # Long: Price crosses below S1 (support) in uptrend with volume
            if (close[i] < s1_aligned[i] and 
                close[i] > ema_200_1d_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price crosses above R1 (resistance) in downtrend with volume
            elif (close[i] > r1_aligned[i] and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to previous day's close (pivot point)
            close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_prev)
            if position == 1:
                if close[i] >= close_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] <= close_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_1dTrend_Volume_Session"
timeframe = "12h"
leverage = 1.0