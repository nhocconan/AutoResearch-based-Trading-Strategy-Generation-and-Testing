#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume_SignalStrength
# Hypothesis: Use signal strength based on distance beyond Camarilla levels to scale position size, reducing whipsaws and trade frequency. Stronger breaks = larger positions. Maintains 1d trend filter and volume confirmation for quality. Designed for fewer, higher-quality trades in both bull and bear markets by avoiding small, noisy breakouts.

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_SignalStrength"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's Camarilla levels (R3, S3)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_range = (prev_high_1d - prev_low_1d) * 1.1 / 4
    r3 = prev_close_1d + camarilla_range
    s3 = prev_close_1d - camarilla_range
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (balanced for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Get aligned 1d close for trend filter
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        close_1d_current = close_1d_aligned[i]
        
        trend_up = close_1d_current > ema50_1d_aligned[i]
        trend_down = close_1d_current < ema50_1d_aligned[i]
        
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # Calculate breakout strength
            long_breakout = close[i] - r3_aligned[i]
            short_breakout = s3_aligned[i] - close[i]
            
            # LONG: Close > R3 with strength, 1d uptrend, volume confirmation
            if long_breakout > 0 and trend_up and vol_confirm:
                # Scale position by breakout strength (capped at 0.30)
                strength_factor = min(long_breakout / (r3_aligned[i] * 0.02), 1.0)  # Normalize by 2% of level
                signal_size = 0.15 + (0.15 * strength_factor)  # Range: 0.15 to 0.30
                signals[i] = signal_size
                position = 1
            # SHORT: Close < S3 with strength, 1d downtrend, volume confirmation
            elif short_breakout > 0 and trend_down and vol_confirm:
                # Scale position by breakout strength (capped at 0.30)
                strength_factor = min(short_breakout / (s3_aligned[i] * 0.02), 1.0)  # Normalize by 2% of level
                signal_size = 0.15 + (0.15 * strength_factor)  # Range: 0.15 to 0.30
                signals[i] = -signal_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < S3 (reversal to opposite level) OR weak momentum
            exit_condition = close[i] < s3_aligned[i]
            weak_momentum = close[i] < r3_aligned[i]  # Below entry level
            if exit_condition or weak_momentum:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position, optionally scale with strength
                long_breakout = close[i] - r3_aligned[i]
                strength_factor = min(long_breakout / (r3_aligned[i] * 0.02), 1.0)
                signal_size = 0.15 + (0.15 * strength_factor)
                signals[i] = signal_size
        elif position == -1:
            # EXIT SHORT: Close > R3 (reversal to opposite level) OR weak momentum
            exit_condition = close[i] > r3_aligned[i]
            weak_momentum = close[i] > s3_aligned[i]  # Above entry level
            if exit_condition or weak_momentum:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position, optionally scale with strength
                short_breakout = s3_aligned[i] - close[i]
                strength_factor = min(short_breakout / (s3_aligned[i] * 0.02), 1.0)
                signal_size = 0.15 + (0.15 * strength_factor)
                signals[i] = -signal_size
    
    return signals