#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 Breakout + 1d Trend + Volume Spike
# Uses daily Camarilla levels from prior day for mean reversion breakouts.
# Enters on break of R1 (long) or S1 (short) with volume confirmation and 1d trend filter.
# Exits when price returns to pivot point or trend reverses.
# Targets 20-40 trades per year (~80-160 total over 4 years) to minimize fee drag.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1/12
    # S1 = Pivot - (H - L) * 1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d trend filter: EMA25 slope
    ema25_1d = pd.Series(close_1d).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_slope = ema25_1d[1:] - ema25_1d[:-1]
    ema25_slope = np.concatenate([[0], ema25_slope])
    ema25_aligned = align_htf_to_ltf(prices, df_1d, ema25_1d)
    ema25_slope_aligned = align_htf_to_ltf(prices, df_1d, ema25_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema25_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema25_slope_val = ema25_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above R1 with volume confirmation and 1d uptrend
            if close[i] > r1_val and vol_conf_val and ema25_slope_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume confirmation and 1d downtrend
            elif close[i] < s1_val and vol_conf_val and ema25_slope_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot or trend turns down
            if close[i] < pivot_val or ema25_slope_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot or trend turns up
            if close[i] > pivot_val or ema25_slope_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals