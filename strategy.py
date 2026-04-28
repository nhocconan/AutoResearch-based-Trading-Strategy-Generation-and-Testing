#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels (R3/S3) from daily range act as strong support/resistance in ranging markets. Breakouts above R3 or below S3 with volume spike and aligned 1d EMA34 trend capture explosive moves. Works in both bull and bear by trading breakouts in direction of higher timeframe trend, with volume confirmation reducing false signals. Designed for 20-40 trades/year to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Based on previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    range_val = prev_high - prev_low
    camarilla_r3 = prev_close + (range_val * 1.1 / 2)  # R3 = C + (H-L)*1.1/2
    camarilla_s3 = prev_close - (range_val * 1.1 / 2)  # S3 = C - (H-L)*1.1/2
    
    # Align to lower timeframe (4h) - values from previous day's close
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA34 and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > camarilla_r3_aligned[i]) and volume_spike[i] and uptrend
        short_breakout = (close[i] < camarilla_s3_aligned[i]) and volume_spike[i] and downtrend
        
        # Exit conditions: reverse when opposite breakout occurs
        long_exit = (close[i] < camarilla_s3_aligned[i]) and volume_spike[i]
        short_exit = (close[i] > camarilla_r3_aligned[i]) and volume_spike[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0