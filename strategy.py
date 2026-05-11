#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade breakouts at Camarilla R1/S1 levels with 1d EMA34 trend filter and volume spike confirmation. Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend). Uses 1d trend for direction and volume to confirm institutional interest. Targets 20-30 trades/year to minimize fee drag.
"""

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # === 1d EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Camarilla Levels from Previous 1d Bar ===
    # Calculate from previous day's OHLC (to avoid look-ahead)
    close_1d_shifted = np.roll(close_1d, 1)
    high_1d_shifted = np.roll(df_1d['high'].values, 1)
    low_1d_shifted = np.roll(df_1d['low'].values, 1)
    close_1d_shifted[0] = close_1d[0]  # first day uses same day
    high_1d_shifted[0] = df_1d['high'].values[0]
    low_1d_shifted[0] = df_1d['low'].values[0]
    
    # Camarilla calculation: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d_shifted + (high_1d_shifted - low_1d_shifted) * 1.1 / 12
    camarilla_s1 = close_1d_shifted - (high_1d_shifted - low_1d_shifted) * 1.1 / 12
    
    # Align to 4h timeframe
    camarilla_r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # 2x average volume for institutional confirmation
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA calculation)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_4h[i]) or np.isnan(camarilla_r1_4h[i]) or 
            np.isnan(camarilla_s1_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above R1 in uptrend (price > EMA34) with volume spike
            if (close[i] > camarilla_r1_4h[i] and 
                close[i] > ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Break below S1 in downtrend (price < EMA34) with volume spike
            elif (close[i] < camarilla_s1_4h[i] and 
                  close[i] < ema34_1d_4h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price returns to opposite Camarilla level or trend reverses
            if position == 1:
                if (close[i] < camarilla_s1_4h[i] or  # broke below S1
                    close[i] < ema34_1d_4h[i]):        # trend turned down
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if (close[i] > camarilla_r1_4h[i] or  # broke above R1
                    close[i] > ema34_1d_4h[i]):       # trend turned up
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals