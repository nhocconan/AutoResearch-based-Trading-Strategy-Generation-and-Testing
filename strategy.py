#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels derived from prior 1d OHLC: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
# Breakout above R3 with uptrend EMA34 + volume spike = long
# Breakdown below S3 with downtrend EMA34 + volume spike = short
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 trend filter from prior completed 1d bar
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_shifted = np.roll(ema34_1d, 1)
    ema34_1d_shifted[0] = np.nan
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d_shifted)
    
    # Calculate prior 1d Camarilla R3 and S3 levels (based on completed 1d bar)
    prior_close_1d = np.roll(close_1d, 1)
    prior_high_1d = np.roll(high_1d, 1)
    prior_low_1d = np.roll(low_1d, 1)
    prior_close_1d[0] = np.nan
    prior_high_1d[0] = np.nan
    prior_low_1d[0] = np.nan
    
    camarilla_range = prior_high_1d - prior_low_1d
    r3 = prior_close_1d + 1.1 * camarilla_range
    s3 = prior_close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above R3 AND 1d EMA34 uptrend AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S3 AND 1d EMA34 downtrend AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below EMA34 OR below S3 (reversion to mean)
            if close[i] < ema34_1d_aligned[i] or close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above EMA34 OR above R3 (reversion to mean)
            if close[i] > ema34_1d_aligned[i] or close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals