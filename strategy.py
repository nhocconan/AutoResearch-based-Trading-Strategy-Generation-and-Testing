#!/usr/bin/env python3
name = "12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # ===== 12h Camarilla Pivot Levels (Previous Day) =====
    # Calculate from previous day's OHLC
    # We need to get previous day's data for each 12h bar
    # Since we're on 12h timeframe, each bar covers half a day
    # We'll use the 1d data to calculate pivots for the previous day
    
    # Get 1d data for pivot calculation
    # For each 12h bar, we use the previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R3 = C + (H-L)*1.1/2
    # S3 = C - (H-L)*1.1/2
    # We'll shift these to align with 12h timeframe
    camarilla_r3 = close_1d_for_pivot + (high_1d - low_1d) * 1.1 / 2
    camarilla_s3 = close_1d_for_pivot - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 12h timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Strong volume spike
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + 1d uptrend + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and  # Price above 1d EMA34 (uptrend)
                vol_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S3 + 1d downtrend + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and  # Price below 1d EMA34 (downtrend)
                  vol_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 or closes below 1d EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price breaks above R3 or closes above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals