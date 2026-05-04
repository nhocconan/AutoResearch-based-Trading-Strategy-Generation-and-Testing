#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla pivot levels provide high-probability intraday reversal/breakout points. 1w EMA50 ensures alignment with weekly trend.
# Volume spike (>2x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 30-100 trades over 4 years (7-25/year).

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend direction
    close_1w = pd.Series(df_1w['close'])
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe (completed 1w bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels for 1d timeframe (based on previous day's range)
    # Camarilla R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous day's high/low to avoid look-ahead
    prev_high = pd.Series(high).shift(1)
    prev_low = pd.Series(low).shift(1)
    prev_close = pd.Series(close).shift(1)
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + (1.1 * camarilla_range / 2.0)
    s3 = prev_close - (1.1 * camarilla_range / 2.0)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ema_20[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point OR trend changes OR volume drops
            # Camarilla pivot point = (high+low+close)/3 of previous day
            pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
            if (close[i] < pivot or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point OR trend changes OR volume drops
            pivot = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
            if (close[i] > pivot or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals