#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume spike
# Uses R4/S4 levels for stronger breakouts, reducing false signals. 12h EMA50 ensures alignment with longer-term trend.
# Volume spike >2.0 filters noise. Designed for 15-25 trades/year to avoid fee drag.
name = "4h_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Get 1d data for Camarilla levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align to 4h
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels for current day
    range_ = prev_high_aligned - prev_low_aligned
    # Camarilla R4, S4 (widest bands for stronger breakouts)
    r4 = prev_close_aligned + 1.1 * range_ * 1.1
    s4 = prev_close_aligned - 1.1 * range_ * 1.1
    
    # Volume confirmation - 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(80, 50)  # warmup period
    
    for i in range(start_idx, n):
        if (np.isnan(r4[i]) or np.isnan(s4[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
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
            # Long entry: break above R4 with trend alignment and volume spike
            if (close[i] > r4[i] and 
                close[i] > ema50_12h_aligned[i] and
                vol_ratio[i] > 2.0):
                signals[i] = 0.28
                position = 1
            # Short entry: break below S4 with trend alignment and volume spike
            elif (close[i] < s4[i] and 
                  close[i] < ema50_12h_aligned[i] and
                  vol_ratio[i] > 2.0):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: break below S4 (mean reversion) OR trend fails
            if close[i] < s4[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: break above R4 (mean reversion) OR trend fails
            if close[i] > r4[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals