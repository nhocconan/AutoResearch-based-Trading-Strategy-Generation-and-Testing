#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3S3 breakout with 1d volume spike and 1w trend filter
# Rationale: Camarilla levels act as strong support/resistance; breakouts with volume
# confirm institutional interest; weekly trend filter avoids counter-trend trades.
# Works in bull markets (breakouts continue) and bear (fades at resistance).
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dVolSpike_1wTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We use R3 and S3 as breakout levels
    # For each 4h bar, we need the previous day's HLC
    # We'll compute daily HLC then shift by 1 to get previous day's values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Previous day's HLC (shift by 1)
    prev_high = np.roll(daily_high, 1)
    prev_low = np.roll(daily_low, 1)
    prev_close = np.roll(daily_close, 1)
    # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla R3 and S3
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    
    # 1d volume average (20-period)
    vol_series = pd.Series(df_1d['volume'])
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 4h
    r3_4h = align_htf_to_ltf(prices, df_1d, r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3)
    vol_avg_20_4h = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema20_1w_4h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for indicators
    start_idx = 30  # enough for 20-period averages and roll
    
    for i in range(start_idx, n):
        # Skip if any key value is NaN
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(vol_avg_20_4h[i]) or np.isnan(ema20_1w_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > vol_avg_20_4h[i] * 1.5  # volume spike
        
        if position == 0:
            # Long: break above R3 with volume and above weekly trend
            if high[i] > r3_4h[i] and vol_ok and close[i] > ema20_1w_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and below weekly trend
            elif low[i] < s3_4h[i] and vol_ok and close[i] < ema20_1w_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 or trend reversal
            if close[i] < s3_4h[i] or close[i] < ema20_1w_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 or trend reversal
            if close[i] > r3_4h[i] or close[i] > ema20_1w_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals