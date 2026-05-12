#158401
# 4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Camarilla pivot levels (R1/S1) from 12h act as dynamic support/resistance.
# Breakout above R1 with 12h EMA50 uptrend and volume confirmation triggers long.
# Breakdown below S1 with 12h EMA50 downtrend and volume confirmation triggers short.
# Uses 4h timeframe for execution, 12h for trend/volume filters. Target: 20-50 trades/year.
# Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets.

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close arrays.
    Returns R1, R2, R3, R4, S1, S2, S3, S4 arrays.
    """
    n = len(high)
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    R1 = close + range_hl * 1.1 / 12
    R2 = close + range_hl * 1.1 / 6
    R3 = close + range_hl * 1.1 / 4
    R4 = close + range_hl * 1.1 / 2
    S1 = close - range_hl * 1.1 / 12
    S2 = close - range_hl * 1.1 / 6
    S3 = close - range_hl * 1.1 / 4
    S4 = close - range_hl * 1.1 / 2
    
    return R1, R2, R3, R4, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot, EMA trend, and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels on 12h
    R1_12h, R2_12h, R3_12h, R4_12h, S1_12h, S2_12h, S3_12h, S4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h volume average (20-period) for volume filter
    volume_12h_series = pd.Series(volume_12h)
    vol_avg_20_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and volume average to be ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_12h_aligned[i]) or np.isnan(S1_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current 4h volume > 1.5x 12h volume average
        volume_filter = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: 12h EMA50 direction
        if i > start_idx:
            ema_50_prev = ema_50_12h_aligned[i-1]
            ema_50_curr = ema_50_12h_aligned[i]
            trend_up = ema_50_curr > ema_50_prev
            trend_down = ema_50_curr < ema_50_prev
        else:
            trend_up = True
            trend_down = False
        
        if position == 0:
            # LONG: Break above R1 with uptrend and volume confirmation
            if close[i] > R1_12h_aligned[i] and trend_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with downtrend and volume confirmation
            elif close[i] < S1_12h_aligned[i] and trend_down and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Close below R1 or trend turns down
            if close[i] < R1_12h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above S1 or trend turns up
            if close[i] > S1_12h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals