#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakout with daily trend filter and volume confirmation.
Breakouts at R3 (resistance) for long, S3 (support) for short in direction of daily EMA34 trend.
Volume > 1.5x 20-period MA confirms breakout strength. Designed for low trade frequency (~25/year)
to minimize fee drag while capturing institutional breakout moves in both bull and bear markets.
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
    
    # Get daily data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla uses previous day's range to calculate support/resistance
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels: H4/L4 = close ± 1.1*range/2, H3/L3 = close ± 1.1*range/4, 
    # H2/L2 = close ± 1.1*range/6, H1/L1 = close ± 1.1*range/12
    # We use H3/L3 and H4/L4 (R3/S3 and R4/S4) for breakouts
    r3 = prev_close + 1.1 * prev_range / 4
    s3 = prev_close - 1.1 * prev_range / 4
    r4 = prev_close + 1.1 * prev_range / 2
    s4 = prev_close - 1.1 * prev_range / 2
    
    # Align Camarilla levels to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Daily trend: price above/below EMA34
        daily_uptrend = close[i] > ema_34_1d_aligned[i]
        daily_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Breakout conditions
        breakout_r3 = close[i] > r3_aligned[i]  # Break above R3
        breakdown_s3 = close[i] < s3_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.5 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of daily trend
        long_entry = vol_confirm and daily_uptrend and breakout_r3
        short_entry = vol_confirm and daily_downtrend and breakdown_s3
        
        # Exit logic: opposite breakdown/breakout or trend change
        long_exit = (close[i] < s3_aligned[i]) or (not daily_uptrend)
        short_exit = (close[i] > r3_aligned[i]) or (not daily_downtrend)
        
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

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0