#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d trend filter and volume spike
# Uses proven Camarilla structure from top performers. 1d EMA34 ensures trend alignment.
# Volume spike >2.0 filters false breakouts. Works in bull via R1/S1 breaks, in bear via reversals at R3/S3.
# Target: 12-37 trades/year to avoid fee drag. Discrete sizing 0.25.
name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 12h
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels for current day
    range_ = prev_high_aligned - prev_low_aligned
    # Camarilla R1, S1, R3, S3
    r1 = prev_close_aligned + 1.1 * range_ * 1.0/12
    s1 = prev_close_aligned - 1.1 * range_ * 1.0/12
    r3 = prev_close_aligned + 1.1 * range_ * 1.1/2
    s3 = prev_close_aligned - 1.1 * range_ * 1.1/2
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 34)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: break above R1 with trend alignment and volume spike
            if (close[i] > r1[i] and 
                close[i] > ema34_1d_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short entry: break below S1 with trend alignment and volume spike
            elif (close[i] < s1[i] and 
                  close[i] < ema34_1d_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below S3 (mean reversion) OR trend fails
            if close[i] < s3[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above R3 (mean reversion) OR trend fails
            if close[i] > r3[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals