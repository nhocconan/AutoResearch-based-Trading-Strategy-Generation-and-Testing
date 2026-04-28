#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla R3/S3 levels from daily pivot act as strong support/resistance. 
Breakouts above R3 or below S3 with daily trend filter (price > EMA34) and volume spike (1.5x 20-period average) 
capture explosive moves in both bull and bear markets. The 12h timeframe reduces noise and overtrading, 
while daily trend filter ensures alignment with higher timeframe momentum. Volume confirmation avoids 
false breakouts. Targets 15-25 trades/year per symbol.
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
    
    # Get daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivot (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    rng = high_1d - low_1d
    # Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close_1d + (rng * 1.1 / 2.0)
    camarilla_s3 = close_1d - (rng * 1.1 / 2.0)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily indicators to 12h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.5x 20-period MA on 12h chart
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions: break of Camarilla R3/S3
        breakout_r3 = close[i] > camarilla_r3_aligned[i]
        breakdown_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and breakout_r3
        short_entry = vol_confirm and downtrend and breakdown_s3
        
        # Exit logic: opposite breakout or trend change
        long_exit = breakdown_s3 or (not uptrend)
        short_exit = breakout_r3 or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0