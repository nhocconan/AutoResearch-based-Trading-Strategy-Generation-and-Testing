#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirm_v1
Breakout above Camarilla R1 or below S1 on 4h with volume confirmation (1.5x avg volume).
Trend filter: price above/below 12h EMA100 to avoid counter-trend trades.
Exit when price returns to Camarilla Pivot point or volume drops below average.
Designed to capture institutional breakouts with volume confirmation in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Camarilla Pivot Levels (using previous day's OHLC) ===
    # For 4h chart, we need daily OHLC to calculate Camarilla levels
    # We'll use 1d data to get proper daily OHLC
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla formulas:
    # H = high, L = low, C = close of previous day
    # R4 = C + (H-L)*1.1/2
    # R3 = C + (H-L)*1.1/4
    # R2 = C + (H-L)*1.1/6
    # R1 = C + (H-L)*1.1/12
    # PP = (H+L+C)/3
    # S1 = C - (H-L)*1.1/12
    # S2 = C - (H-L)*1.1/6
    # S3 = C - (H-L)*1.1/4
    # S4 = C - (H-L)*1.1/2
    
    # We need to align daily data to 4h bars
    # Get daily OHLC values
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_pp = (daily_high + daily_low + daily_close) / 3.0
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12.0
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12.0
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Average (20-period) ===
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h EMA100 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    ema_100_12h = pd.Series(df_12h['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(ema_100_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and above 12h EMA100
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > vol_avg[i] * 1.5 and 
                close[i] > ema_100_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1 with volume confirmation and below 12h EMA100
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > vol_avg[i] * 1.5 and 
                  close[i] < ema_100_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price returns to Pivot point OR volume drops below average
            if (close[i] <= camarilla_pp_aligned[i] or 
                volume[i] < vol_avg[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Pivot point OR volume drops below average
            if (close[i] >= camarilla_pp_aligned[i] or 
                volume[i] < vol_avg[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_Volume_Confirm_v1"
timeframe = "4h"
leverage = 1.0