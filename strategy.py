#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1d Camarilla R3 AND price > 1w EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1d Camarilla S3 AND price < 1w EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses 1d Camarilla pivot point (PP) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Camarilla levels from 1d provide intraday support/resistance; 1w EMA34 filters primary trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Calculate 1d Camarilla levels (using prior day's OHLC)
    # We need to resample to 1d to get prior day's OHLC for Camarilla calculation
    # But we cannot resample in loop - so we precompute 1d OHLC using get_htf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: based on prior day's OHLC
    # Camarilla formulas:
    # PP = (high + low + close) / 3
    # R3 = PP + (high - low) * 1.1 / 2
    # S3 = PP - (high - low) * 1.1 / 2
    # We shift by 1 to use prior day's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Shift to get prior day's levels (today's trading uses yesterday's Camarilla)
    pp_1d = np.roll(pp_1d, 1)
    r3_1d = np.roll(r3_1d, 1)
    s3_1d = np.roll(s3_1d, 1)
    pp_1d[0] = np.nan
    r3_1d[0] = np.nan
    s3_1d[0] = np.nan
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_12h = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1w EMA34, volume confirmation
            if close[i] > r3_12h[i] and close[i] > ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1w EMA34, volume confirmation
            elif close[i] < s3_12h[i] and close[i] < ema34_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla pivot point OR volume drops below average
            if close[i] < pp_12h[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla pivot point OR volume drops below average
            if close[i] > pp_12h[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals