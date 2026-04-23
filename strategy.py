#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
- Camarilla pivot levels calculated from prior 1d OHLC
- Long: Close breaks above R3 (resistance 3) + price > 1d EMA50 + volume > 2.0x 24-period avg
- Short: Close breaks below S3 (support 3) + price < 1d EMA50 + volume > 2.0x 24-period avg
- Exit: Close returns to the prior day's pivot point (PP)
- Uses Camarilla for precise intraday levels, 1d EMA50 for HTF trend filter, volume for confirmation
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
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
    
    # Volume confirmation: > 2.0x 24-period average (24 * 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate prior 1d Camarilla levels (need prior day's OHLC)
    # We'll compute these for each 1d bar then align to 12h
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on prior day's range
    # R4 = Close + 1.5*(High-Low), R3 = Close + 1.25*(High-Low), etc.
    # S4 = Close - 1.5*(High-Low), S3 = Close - 1.25*(High-Low)
    # PP = (High + Low + Close)/3
    rang = high_1d - low_1d
    r3 = close_1d + 1.25 * rang
    s3 = close_1d - 1.25 * rang
    pp = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 12h timeframe (prior day's levels available at 12h open)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # Need 50 for EMA50, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pp_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + price > 1d EMA50 + volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + price < 1d EMA50 + volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close returns to pivot point (PP) or below
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close returns to pivot point (PP) or above
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA50_VolumeConfirm"
timeframe = "12h"
leverage = 1.0