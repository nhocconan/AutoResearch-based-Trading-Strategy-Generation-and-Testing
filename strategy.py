#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot (R3/S3) breakout with 1-day volume spike and 1-day EMA34 trend filter.
Only enter long when price breaks above R3 with volume spike and daily EMA34 up; short when price breaks below S3 with volume spike and daily EMA34 down.
Exit on break of opposite Camarilla level (S3 for long, R3 for short) or loss of EMA34 trend.
Uses actual Camarilla formula based on prior day's range. Designed for low trade frequency by requiring confluence of price level break, volume confirmation, and trend alignment.
Works in both bull and bear markets by following daily EMA34 trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla levels, EMA, and volume - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    #          S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R3 and S3 as primary breakout levels
    daily_close = df_daily['close'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    
    # Calculate Camarilla R3 and S3 for each day
    camarilla_r3 = np.full_like(daily_close, np.nan)
    camarilla_s3 = np.full_like(daily_close, np.nan)
    
    for i in range(1, len(daily_close)):
        if i-1 >= 0 and not (np.isnan(daily_high[i-1]) or np.isnan(daily_low[i-1]) or np.isnan(daily_close[i-1])):
            rng = daily_high[i-1] - daily_low[i-1]
            camarilla_r3[i] = daily_close[i-1] + rng * 1.1 / 4
            camarilla_s3[i] = daily_close[i-1] - rng * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_s3)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily volume spike: current day's volume > 2.0x 20-day average
    daily_volume = df_daily['volume'].values
    vol_ma_20_daily = pd.Series(daily_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_daily = daily_volume > 2.0 * vol_ma_20_daily
    vol_spike_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_spike_daily.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_spike_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition from daily data
        vol_spike = vol_spike_daily_aligned[i] > 0.5  # True if spike
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + daily EMA34 up
            if close[i] > camarilla_r3_aligned[i] and vol_spike and ema34_daily_aligned[i] > ema34_daily_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + volume spike + daily EMA34 down
            elif close[i] < camarilla_s3_aligned[i] and vol_spike and ema34_daily_aligned[i] < ema34_daily_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Camarilla level or loss of EMA34 trend
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S3 or EMA34 turns down
                if close[i] < camarilla_s3_aligned[i] or ema34_daily_aligned[i] < ema34_daily_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R3 or EMA34 turns up
                if close[i] > camarilla_r3_aligned[i] or ema34_daily_aligned[i] > ema34_daily_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0