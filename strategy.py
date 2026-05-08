#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R3/S3 breakout with volume confirmation and 1w trend filter
# Long when price breaks above R3 with volume > 1.5x 20-period avg and 1w EMA21 upward
# Short when price breaks below S3 with volume > 1.5x 20-period avg and 1w EMA21 downward
# Exit when price crosses opposite pivot level (S3 for long, R3 for short) or trend reverses
# Camarilla provides institutional levels, volume confirms breakout strength, weekly filter avoids counter-trend
# Targets 25-40 trades per year for optimal fee drag (~100-160 total over 4 years)

name = "4h_Camarilla_R3S3_Breakout_1wEMA21_Trend"
timeframe = "4h"
leverage = 1.0

def calculate_pivot_points(high, low, close):
    """Calculate Camarilla pivot levels"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for each bar using previous day's data
    # We need to use previous day's HLC for today's levels
    r3_levels = np.full(n, np.nan)
    s3_levels = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's data to calculate today's levels
        r3, s3 = calculate_pivot_points(high[i-1], low[i-1], close[i-1])
        r3_levels[i] = r3
        s3_levels[i] = s3
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate EMA21 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_slope = ema21_1w[1:] - ema21_1w[:-1]  # slope: positive = uptrend
    ema21_1w_slope = np.concatenate([[0], ema21_1w_slope])  # align length
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    ema21_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 21  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_levels[i]) or np.isnan(s3_levels[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(ema21_1w_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        r3_val = r3_levels[i]
        s3_val = s3_levels[i]
        ema21_val = ema21_1w_aligned[i]
        ema21_slope = ema21_1w_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above R3, volume confirmation, 1w uptrend (positive slope)
            if close_val > r3_val and vol_conf_val and ema21_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3, volume confirmation, 1w downtrend (negative slope)
            elif close_val < s3_val and vol_conf_val and ema21_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S3 or 1w trend turns down
            if close_val < s3_val or ema21_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R3 or 1w trend turns up
            if close_val > r3_val or ema21_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals