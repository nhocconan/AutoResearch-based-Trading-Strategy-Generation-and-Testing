#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S3_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot levels and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 10:
        return np.zeros(n)
    
    # Previous week's OHLC for weekly pivot calculation
    prev_high = df_w['high'].shift(1).values
    prev_low = df_w['low'].shift(1).values
    prev_close = df_w['close'].shift(1).values
    
    # Weekly pivot point
    pp = (prev_high + prev_low + prev_close) / 3.0
    
    # Weekly R3 and S3 levels (more extreme than R1/S1)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    
    # Weekly trend filter: EMA34 on weekly close
    ema34_w = pd.Series(df_w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current week volume > 1.5 * 10-week average
    vol_series = pd.Series(df_w['volume'].values)
    vol_ma = vol_series.rolling(window=10, min_periods=10).mean().values
    volume_filter_w = df_w['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h
    r3_6h = align_htf_to_ltf(prices, df_w, r3)
    s3_6h = align_htf_to_ltf(prices, df_w, s3)
    ema34_w_6h = align_htf_to_ltf(prices, df_w, ema34_w)
    volume_filter_6h = align_htf_to_ltf(prices, df_w, volume_filter_w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 10)  # Need enough data for weekly EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema34_w_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        trend = ema34_w_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: break above R3 with volume and above weekly trend
            if close[i] > r3_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume and below weekly trend
            elif close[i] < s3_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (mean reversion to center)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to center)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals