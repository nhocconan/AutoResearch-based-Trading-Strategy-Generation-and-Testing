#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla pivot (R3/S3) breakout with 1-day trend filter and volume confirmation.
Trades only on breakouts of key pivot levels in the direction of the daily trend.
Designed to work in both bull and bear markets by using the 1-day trend as filter.
Target: 15-35 trades/year per symbol (60-140 total over 4 years) to minimize fee drag.
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
    
    # Get 1-day data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from previous day
    # Using typical price: (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels: R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    # where PP = typical price
    pp = typical_price.values
    camarilla_r3 = pp + range_hl * 1.1 / 2
    camarilla_s3 = pp - range_hl * 1.1 / 2
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan
    camarilla_s3[0] = np.nan
    
    # Calculate 1-day EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12-hour volume for confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need pivots, EMA, and volume MA
    start_idx = max(30, 34, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        trend_1d = ema_34_1d_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_12h_aligned[i]
        
        # Volume filter: volume > 1.8x 12h average (selective to reduce trades)
        vol_filter = vol_now > 1.8 * vol_ma
        
        # Entry conditions: Camarilla level breakout with volume and 1d trend alignment
        if position == 0:
            # Long: break above R3 + volume + 1d uptrend
            if close[i] > r3_level and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: break below S3 + volume + 1d downtrend
            elif close[i] < s3_level and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 1-day EMA or S3 level (stop and reverse)
            if close[i] < trend_1d or close[i] < s3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 1-day EMA or R3 level (stop and reverse)
            if close[i] > trend_1d or close[i] > r3_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0