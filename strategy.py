#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points (R2/S2) for breakout direction with
# daily EMA200 trend filter and volume confirmation. Weekly pivots provide stable
# support/resistance levels that work in both bull and bear markets by aligning with
# higher timeframe structure. Target: 60-120 total trades over 4 years (15-30/year)
# to minimize fee drag while capturing significant moves.

name = "6h_WeeklyPivot_R2S2_Breakout_DailyTrend_Volume"
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
    
    # Get weekly data for pivot points
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 5:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 200:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    pivot_w = (high_w + low_w + close_w) / 3
    r2_w = pivot_w + (high_w - low_w)
    s2_w = pivot_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe
    r2_w_aligned = align_htf_to_ltf(prices, df_w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_w, s2_w)
    
    # Daily EMA200 for trend filter
    close_d = df_d['close'].values
    ema200_d = pd.Series(close_d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_d_aligned = align_htf_to_ltf(prices, df_d, ema200_d)
    
    # Volume spike filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r2_w_aligned[i]) or 
            np.isnan(s2_w_aligned[i]) or
            np.isnan(ema200_d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r2 = r2_w_aligned[i]
        s2 = s2_w_aligned[i]
        ema200 = ema200_d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above R2 + daily uptrend + volume spike
            if close[i] > r2 and close[i] > ema200 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below S2 + daily downtrend + volume spike
            elif close[i] < s2 and close[i] < ema200 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below S2 or daily trend turns down
            if close[i] < s2 or close[i] < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above R2 or daily trend turns up
            if close[i] > r2 or close[i] > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals