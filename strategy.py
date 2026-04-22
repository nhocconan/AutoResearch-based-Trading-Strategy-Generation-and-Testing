#!/usr/bin/env python3

"""
Hypothesis: 12-hour Camarilla pivot breakout with daily trend filter and volume spike confirmation.
Long when price breaks above daily Camarilla R3 during daily uptrend with volume spike.
Short when price breaks below daily Camarilla S3 during daily downtrend with volume spike.
Exit when price returns to daily Camarilla C level or daily trend reverses.
Designed for low trade frequency (15-30 trades/year) by requiring daily trend alignment and volume spikes.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivots and trend - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = (high - low)
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # C = close
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + range_1d * 1.1 / 2
    camarilla_s3 = close_1d - range_1d * 1.1 / 2
    camarilla_c = close_1d  # pivot point
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)
    
    # Daily trend: 34-period EMA on daily close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_c_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_30[i]
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume spike
            if close[i] > camarilla_r3_aligned[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < camarilla_s3_aligned[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to C level or daily trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below C level or daily trend turns down
                if close[i] < camarilla_c_aligned[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above C level or daily trend turns up
                if close[i] > camarilla_c_aligned[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0