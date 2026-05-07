#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Fade_1dTrend_Volume
# Hypothesis: Fade at Camarilla R3/S3 levels on 6h with 1d EMA trend filter and volume spike confirmation
# Works in bull markets by fading overextended rallies at R3, in bear markets by fading panic drops at S3
# Trend filter ensures we only fade against the 1d trend (sell strength in uptrend, buy weakness in downtrend)
# Volume spike confirms institutional interest at these key levels
# Target: 50-150 total trades over 4 years (~12-37/year) with position size 0.25

name = "6h_Camarilla_R3_S3_Fade_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's range
    # Using 1d high/low/close from previous day
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    R3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low) * 1.1 / 12
    S3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low) * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume ratio: current volume / 24-period average volume (approx 6d on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need 24 periods for volume MA and 1d data shifted
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Fade conditions: price at R3/S3 with volume spike
        at_R3 = close[i] >= R3_aligned[i] * 0.999  # Within 0.1% of R3
        at_S3 = close[i] <= S3_aligned[i] * 1.001  # Within 0.1% of S3
        volume_confirm = vol_ratio[i] > 2.0  # Volume spike > 2x average
        
        # Trend filter from 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Fade R3: sell when price reaches R3 in uptrend
            if at_R3 and volume_confirm and uptrend:
                signals[i] = -0.25
                position = -1
            # Fade S3: buy when price reaches S3 in downtrend
            elif at_S3 and volume_confirm and downtrend:
                signals[i] = 0.25
                position = 1
        elif position == 1:
            # Exit long: price reaches midpoint or trend reversal
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] >= midpoint or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches midpoint or trend reversal
            midpoint = (S3_aligned[i] + R3_aligned[i]) / 2
            if close[i] <= midpoint or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals