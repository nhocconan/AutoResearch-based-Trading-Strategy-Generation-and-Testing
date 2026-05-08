#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla Pivot (S3/R3) breakout with 1-day trend filter and volume spike
# Long when price breaks above R3 + daily EMA(34) uptrend + volume spike
# Short when price breaks below S3 + daily EMA(34) downtrend + volume spike
# Camarilla levels provide precise intraday support/resistance, effective in ranging and trending markets
# Daily trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data once for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels (S3, R3) from previous day's range
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close_val = df_1d['close'].values
    camarilla_r3 = daily_close_val + 1.1 * (daily_high - daily_low) / 2
    camarilla_s3 = daily_close_val - 1.1 * (daily_high - daily_low) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above R3 + daily uptrend + volume spike
            if close[i] > r3 and close[i] > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < s3 and close[i] < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below R3 OR daily trend turns down
            if close[i] < r3 or close[i] < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above S3 OR daily trend turns up
            if close[i] > s3 or close[i] > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals