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
    
    # Get 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d weekly pivot points from previous week
    # Weekly pivot: (weekly high + weekly low + weekly close) / 3
    # We'll use rolling window of 5 days (1 week) to calculate weekly levels
    weekly_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
    
    # Calculate pivot points
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r2 + (weekly_high - weekly_low))  # R4 = R3 + (R2-R1)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s2 - (weekly_high - weekly_low))  # S4 = S3 - (S2-S1)
    
    # Calculate 6-period RSI for overbought/oversold signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Fade at extreme levels (R3/S3) with volume confirmation
        fade_long = (close[i] <= s3_aligned[i]) and (rsi[i] < 30) and vol_filter[i]
        fade_short = (close[i] >= r3_aligned[i]) and (rsi[i] > 70) and vol_filter[i]
        
        # Breakout continuation at extreme levels (R4/S4) with volume confirmation
        breakout_long = (close[i] >= r4_aligned[i]) and (rsi[i] > 50) and vol_filter[i]
        breakout_short = (close[i] <= s4_aligned[i]) and (rsi[i] < 50) and vol_filter[i]
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = (close[i] >= pivot_aligned[i]) or (close[i] <= s1_aligned[i]) if not np.isnan(s1_aligned[i]) else False
        short_exit = (close[i] <= pivot_aligned[i]) or (close[i] >= r1_aligned[i]) if not np.isnan(r1_aligned[i]) else False
        
        if (fade_long or breakout_long) and position != 1:
            position = 1
            signals[i] = 0.25
        elif (fade_short or breakout_short) and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_weekly_pivot_r3s3_r4s4_volume_filter_v1"
timeframe = "6h"
leverage = 1.0