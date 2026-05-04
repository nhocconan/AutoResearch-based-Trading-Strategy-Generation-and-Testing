#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla pivot levels identify intraday support/resistance. Breakout of R3/S3 with 1d EMA trend alignment
# captures strong momentum moves. Volume confirmation (>2x 20 EMA) ensures institutional participation.
# Discrete sizing 0.25 limits risk. Works in bull/bear via trend filter. Target: 75-200 trades over 4 years.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels for 4h timeframe using previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), etc.
    # We need previous day's OHLC to calculate today's levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC from 4h data (resample to 1d using actual Binance 1d boundaries)
    # Since we can't resample, we'll use the 1d data we already fetched for OHLC
    # Camarilla levels are based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values  # Previous day's close
    prev_high = df_1d['high'].shift(1).values    # Previous day's high
    prev_low = df_1d['low'].shift(1).values      # Previous day's low
    
    # Align previous day's OHLC to 4h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Calculate Camarilla levels: R3 = prev_close + 1.25*(prev_high - prev_low)
    #                        S3 = prev_close - 1.25*(prev_high - prev_low)
    camarilla_range = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + 1.25 * camarilla_range
    s3 = prev_close_aligned - 1.25 * camarilla_range
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirm = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + uptrend + volume spike
            if close[i] > r3[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + downtrend + volume spike
            elif close[i] < s3[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point (close) OR trend changes OR volume drops
            # Pivot point = (prev_high + prev_low + prev_close) / 3
            pp = (prev_high_aligned[i] + prev_low_aligned[i] + prev_close_aligned[i]) / 3.0
            if (close[i] < pp or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point (close) OR trend changes OR volume drops
            pp = (prev_high_aligned[i] + prev_low_aligned[i] + prev_close_aligned[i]) / 3.0
            if (close[i] > pp or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals