#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Camarilla levels identify key intraday support/resistance. Breakouts above R3 or below S3 with
# 1d EMA34 trend alignment capture strong momentum moves. Volume spike (>1.8x 20 EMA) confirms
# participation. Discrete sizing 0.25 limits risk. Works in bull/bear via trend filter.
# Target: 75-200 trades over 4 years (19-50/year) to avoid fee drag.

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
    open_ = prices['open'].values  # Camarilla uses OHLC
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe (completed 1d bar only)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We'll compute daily OHLC then shift to avoid look-ahead
    df_1d['prev_close'] = df_1d['close'].shift(1)
    df_1d['prev_high'] = df_1d['high'].shift(1)
    df_1d['prev_low'] = df_1d['low'].shift(1)
    df_1d['prev_open'] = df_1d['open'].shift(1)
    
    # Typical price for Camarilla calculation
    typical_price = (df_1d['prev_high'] + df_1d['prev_low'] + df_1d['prev_close']) / 3.0
    range_hl = df_1d['prev_high'] - df_1d['prev_low']
    
    # Camarilla levels
    r3 = typical_price + 1.1 * range_hl
    s3 = typical_price - 1.1 * range_hl
    r3_vals = r3.values
    s3_vals = s3.values
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8 x 20-period EMA
        volume_confirm = volume[i] > (1.8 * vol_ema_20[i])
        
        if position == 0:
            # Long conditions: price breaks above R3 + uptrend + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3 + downtrend + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to midpoint of R3/S3 OR trend changes OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] < midpoint or 
                close[i] < ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to midpoint of R3/S3 OR trend changes OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] > midpoint or 
                close[i] > ema34_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals