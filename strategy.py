#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ATR-based volatility breakout with weekly trend filter and volume confirmation
# Uses weekly ATR(14) scaled by 0.5 to set breakout levels from prior week's close
# Weekly trend filter: price above/below weekly EMA(34)
# Volume confirmation: current volume > 1.5x 20-period average
# Designed for low trade frequency (target: 20-50 total trades over 4 years = 5-12/year)
# Works in both bull and bear markets by capturing volatility expansions aligned with weekly trend

name = "6h_VolatilityBreakout_WeeklyTrend_Volume"
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
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly ATR(14) for volatility
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = np.roll(close_1w, 1)
    close_1w_shift[0] = np.nan
    tr1 = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - close_1w_shift), np.abs(low_1w - close_1w_shift)))
    atr14_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Calculate weekly close for breakout levels
    weekly_close = df_1w['close'].values
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(atr14_1w_aligned[i]) or 
            np.isnan(weekly_close_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1w_val = ema34_1w_aligned[i]
        atr14_1w_val = atr14_1w_aligned[i]
        weekly_close_val = weekly_close_aligned[i]
        vol_spike = volume_spike[i]
        
        # Calculate breakout levels: weekly close ± (0.5 * weekly ATR)
        upper_break = weekly_close_val + (0.5 * atr14_1w_val)
        lower_break = weekly_close_val - (0.5 * atr14_1w_val)
        
        if position == 0:
            # Enter long: price breaks above upper level + uptrend + volume spike
            if (close[i] > upper_break and 
                close[i] > ema34_1w_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower level + downtrend + volume spike
            elif (close[i] < lower_break and 
                  close[i] < ema34_1w_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly close OR trend turns down
            if (close[i] < weekly_close_val or close[i] < ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly close OR trend turns up
            if (close[i] > weekly_close_val or close[i] > ema34_1w_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals