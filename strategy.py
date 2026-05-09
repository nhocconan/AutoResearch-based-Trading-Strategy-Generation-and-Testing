#!/usr/bin/env python3
# Hypothesis: 6h price action relative to daily pivot zones with volume and trend confirmation
# Uses daily pivot points (PP, R1, S1, R2, S2) to identify key support/resistance zones
# Long when price breaks above R1 with volume confirmation and is above daily EMA50 (uptrend)
# Short when price breaks below S1 with volume confirmation and is below daily EMA50 (downtrend)
# Exit when price returns to the daily pivot zone (between S1 and R1) or trend reverses
# Designed to capture breakouts from key daily levels while avoiding false breakouts
# Works in both bull and bear markets by following the daily trend filter

name = "6h_Pivot_R1S1_Breakout_Trend_Volume"
timeframe = "6h"
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
    
    # Daily EMA50 for trend filter (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily pivot points: (H + L + C) / 3
    pp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Support and resistance levels
    r1_1d = 2 * pp_1d - df_1d['low']
    s1_1d = 2 * pp_1d - df_1d['high']
    r2_1d = pp_1d + (df_1d['high'] - df_1d['low'])
    s2_1d = pp_1d - (df_1d['high'] - df_1d['low'])
    
    # Align daily indicators to 6h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d.values)
    
    # Volume confirmation: current volume > 1.8x 20-period average (higher threshold for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 with volume and uptrend
            if (close[i] > r1_aligned[i] and 
                close[i] > ema50_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema50_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot zone (between S1 and R1) or trend turns bearish
            if (close[i] <= r1_aligned[i]) or (close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot zone (between S1 and R1) or trend turns bullish
            if (close[i] >= s1_aligned[i]) or (close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals