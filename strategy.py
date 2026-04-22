#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla Pivot Reversal with Daily Trend and Volume Spike.
Long when price touches S3/S2 with bullish reversal and daily EMA34 uptrend with volume spike.
Short when price touches R3/R2 with bearish reversal and daily EMA34 downtrend with volume spike.
Uses Camarilla levels from prior day for reversal entries in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla levels and trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # R2 = close + 0.6 * (high - low)
    # R1 = close + 0.382 * (high - low)
    # S1 = close - 0.382 * (high - low)
    # S2 = close - 0.6 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    daily_range = high_1d - low_1d
    r3 = close_1d + 1.1 * daily_range
    r2 = close_1d + 0.6 * daily_range
    s2 = close_1d - 0.6 * daily_range
    s3 = close_1d - 1.1 * daily_range
    
    # Align Camarilla levels to 12h timeframe (previous day's levels for current day trading)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price at S3/S2 with bullish reversal, daily EMA34 up, volume spike
            if ((close[i] <= s3_aligned[i] * 1.002 and close[i] >= s3_aligned[i] * 0.998) or
                (close[i] <= s2_aligned[i] * 1.002 and close[i] >= s2_aligned[i] * 0.998)):
                # Bullish reversal: current close > prior close
                if i > 0 and close[i] > close[i-1] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                    signals[i] = 0.25
                    position = 1
            # Short: Price at R3/R2 with bearish reversal, daily EMA34 down, volume spike
            elif ((close[i] >= r3_aligned[i] * 0.998 and close[i] <= r3_aligned[i] * 1.002) or
                  (close[i] >= r2_aligned[i] * 0.998 and close[i] <= r2_aligned[i] * 1.002)):
                # Bearish reversal: current close < prior close
                if i > 0 and close[i] < close[i-1] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: Price moves away from pivot level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price moves above S2 or trend turns down
                if (close[i] > s2_aligned[i] * 1.01 or 
                    ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price moves below R2 or trend turns up
                if (close[i] < r2_aligned[i] * 0.99 or 
                    ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_Pivot_Reversal_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0