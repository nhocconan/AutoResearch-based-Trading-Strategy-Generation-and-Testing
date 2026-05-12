#!/usr/bin/env python3
"""
12h Camarilla Pivot R3/S3 Breakout + Daily Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) from daily charts act as strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and daily trend alignment capture
significant moves in both bull and bear markets. Designed for low trade frequency (~20/year)
on 12h timeframe to minimize fee decay while capturing sustained trends.
"""
name = "12h_Camarilla_R3S3_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # === Daily Camarilla Pivot Levels (R3, S3) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each daily bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use R3 and S3 as breakout levels
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align to 12h timeframe with 1-day delay (need previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=1)
    
    # === Daily EMA34 for trend direction ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above daily R3 + daily uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below daily S3 + daily downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below daily S3 OR price crosses below daily EMA34
            if close[i] < s3_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above daily R3 OR price crosses above daily EMA34
            if close[i] > r3_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals