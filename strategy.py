#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance on daily timeframe. 
Breakout above R3 with volume spike and weekly uptrend = long. Breakdown below S3 with volume spike and weekly downtrend = short.
Uses weekly trend filter to avoid counter-trend trades. Target: 15-25 trades/year per symbol.
Timeframe: 1d, HTF: 1w
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter (more responsive than EMA50)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily pivot points for Camarilla levels (using previous day's OHLC)
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # R3 = Close + 1.1*(High-Low)/2
    # S3 = Close - 1.1*(High-Low)/2
    # Using previous day's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # Set first value to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Volume spike detector (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need at least 1 previous day)
    start_idx = 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and weekly uptrend
            if close[i] > camarilla_r3[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and weekly downtrend
            elif close[i] < camarilla_s3[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters below R3 OR weekly trend changes to downtrend
            if close[i] < camarilla_r3[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters above S3 OR weekly trend changes to uptrend
            if close[i] > camarilla_s3[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0