#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Camarilla pivots identify key intraday support/resistance levels. R3/S3 breaks signal strong momentum.
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend entries.
# Volume spike (>1.5x 20 EMA) confirms participation. Discrete sizing 0.20 limits risk and fees.
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
# Session filter (08-20 UTC) reduces noise and overtrading.

name = "1h_Camarilla_R3S3_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:  # Need at least 1 bar for OHLC
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar
    # Typical price = (H+L+C)/3
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3.0
    # Camarilla width = (H-L) * 1.1 / 12
    camarilla_width = (df_4h['high'] - df_4h['low']) * 1.1 / 12.0
    
    # R3 = C + (H-L) * 1.1/12 * 4 = typical_price + camarilla_width * 4
    # S3 = C - (H-L) * 1.1/12 * 4 = typical_price - camarilla_width * 4
    r3_levels = typical_price + camarilla_width * 4.0
    s3_levels = typical_price - camarilla_width * 4.0
    
    # Al Camarilla levels to 1h timeframe (completed 4h bar only)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3_levels.values)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3_levels.values)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3/S3 OR trend changes OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to midpoint of R3/S3 OR trend changes OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals