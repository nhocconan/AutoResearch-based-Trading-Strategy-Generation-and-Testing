#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d EMA50 trend filter + weekly Camarilla pivot breakout + volume confirmation.
Long when price breaks above weekly Camarilla R4 level with 1d EMA50 rising and volume > 1.3x 20-period 1d volume average.
Short when price breaks below weekly Camarilla S4 level with 1d EMA50 falling and volume > 1.3x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Weekly pivots provide structural levels; EMA50 filters trend alignment; volume confirms participation.
Designed to work in bull markets (breakout continuation) and bear markets (mean reversion after volatility expansion).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_prev = np.roll(ema_50_1d, 1)
    ema_50_1d_prev[0] = np.nan
    ema_50_1d_rising = ema_50_1d > ema_50_1d_prev
    ema_50_1d_falling = ema_50_1d < ema_50_1d_prev
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly Camarilla levels
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = close_1w + ((high_1w - low_1w) * 1.1 / 2)
    camarilla_s4 = close_1w - ((high_1w - low_1w) * 1.1 / 2)
    
    # Align all to 6h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_50_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_rising)
    ema_50_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_falling)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema_50_1d_rising_aligned[i]) or np.isnan(ema_50_1d_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Camarilla R4 with rising 1d EMA50 and volume
            if (close[i] > camarilla_r4_aligned[i] and 
                ema_50_1d_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Camarilla S4 with falling 1d EMA50 and volume
            elif (close[i] < camarilla_s4_aligned[i] and 
                  ema_50_1d_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Camarilla R3 level
            camarilla_r3 = close_1w + ((high_1w - low_1w) * 1.1/4)
            camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
            if close[i] < camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Camarilla S3 level
            camarilla_s3 = close_1w - ((high_1w - low_1w) * 1.1/4)
            camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
            if close[i] > camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dEMA50_1wCamarilla_S4R4_Volume_Confirm"
timeframe = "6h"
leverage = 1.0