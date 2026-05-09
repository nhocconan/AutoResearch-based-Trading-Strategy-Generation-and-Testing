#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    """
    12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
    - Uses daily trend filter to avoid counter-trend trades
    - R3/S3 breakouts for momentum entries
    - Volume spike filter to avoid false breakouts
    - Target: 12-37 trades/year on 12h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 from previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    prev_daily_range = prev_high_1d - prev_low_1d
    pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r3 = pivot + 1.1 * prev_daily_range * 1.05  # R3 = pivot + 1.1 * range * 1.05
    s3 = pivot - 1.1 * prev_daily_range * 1.05  # S3 = pivot - 1.1 * range * 1.05
    
    # Align Camarilla levels to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (20-period for 12h)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(ema34_12h[i]) or 
            np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Break above Camarilla R3 with uptrend on 1d, volume spike
            if (close[i] > r3_12h[i] and close[i] > ema34_12h[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 with downtrend on 1d, volume spike
            elif (close[i] < s3_12h[i] and close[i] < ema34_12h[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below EMA34
            if close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above EMA34
            if close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals