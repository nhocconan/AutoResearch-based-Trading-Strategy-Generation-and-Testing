#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA21 trend filter and volume spike confirmation
# Camarilla R3/S3 levels represent stronger breakout zones than R1/S1, reducing false signals
# Breakout above R3 with bullish 1w EMA21 trend and volume spike = long
# Breakdown below S3 with bearish 1w EMA21 trend and volume spike = short
# Uses 1d primary timeframe targeting 30-100 total trades over 4 years (7-25/year)
# 1w trend filter provides robust bull/bear market adaptation
# Volume confirmation ensures breakout legitimacy
# Discrete position sizing (0.30) minimizes fee churn

name = "1d_Camarilla_R3S3_Breakout_1wEMA21_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get 1w data for Camarilla pivot calculation and EMA21 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R3 = Close + 1.1*(High-Low)/6, S3 = Close - 1.1*(High-Low)/6
    camarilla_range = high_1w - low_1w
    r3 = close_1w + (1.1 * camarilla_range / 6)
    s3 = close_1w - (1.1 * camarilla_range / 6)
    
    # Align to 1d timeframe (wait for completed 1w bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1w EMA21 trend filter from prior completed 1w bar
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_shifted = np.roll(ema21_1w, 1)
    ema21_1w_shifted[0] = np.nan
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 AND 1w EMA21 uptrend AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema21_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: break below S3 AND 1w EMA21 downtrend AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema21_1w_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below R3 OR below 1w EMA21
            if close[i] < r3_aligned[i] or close[i] < ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above S3 OR above 1w EMA21
            if close[i] > s3_aligned[i] or close[i] > ema21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals