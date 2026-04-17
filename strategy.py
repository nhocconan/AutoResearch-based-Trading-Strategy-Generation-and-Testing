#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d EMA200 trend filter + daily Camarilla pivot (R1/S1) breakout + volume confirmation.
Long when price breaks above daily Camarilla R1 level with 1d EMA200 rising and volume > 1.5x 20-period 1d volume average.
Short when price breaks below daily Camarilla S1 level with 1d EMA200 falling and volume > 1.5x 20-period 1d volume average.
Uses discrete position sizing 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
Daily pivots provide intraday structure; EMA200 filters primary trend; volume confirms institutional participation.
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
    
    # Get 1d data for EMA200 trend and volume
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_prev = np.roll(ema_200_1d, 1)
    ema_200_1d_prev[0] = np.nan
    ema_200_1d_rising = ema_200_1d > ema_200_1d_prev
    ema_200_1d_falling = ema_200_1d < ema_200_1d_prev
    
    # Calculate 1d volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily Camarilla levels
    # Camarilla: R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    camarilla_r1 = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1 = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align all to 4h
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_200_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d_rising)
    ema_200_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d_falling)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_200_1d_rising_aligned[i]) or np.isnan(ema_200_1d_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above daily Camarilla R1 with rising 1d EMA200 and volume
            if (close[i] > camarilla_r1_aligned[i] and 
                ema_200_1d_rising_aligned[i] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below daily Camarilla S1 with falling 1d EMA200 and volume
            elif (close[i] < camarilla_s1_aligned[i] and 
                  ema_200_1d_falling_aligned[i] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below daily Camarilla S1 level
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above daily Camarilla R1 level
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dEMA200_1dCamarilla_R1S1_Volume_Confirm"
timeframe = "4h"
leverage = 1.0