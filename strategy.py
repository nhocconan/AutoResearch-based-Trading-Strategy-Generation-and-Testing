#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d EMA34 trend + volume spike
# Long when price breaks above 4h Camarilla R3 AND 1d EMA34 > price (uptrend) AND volume spike
# Short when price breaks below 4h Camarilla S3 AND 1d EMA34 < price (downtrend) AND volume spike
# Exit when price retouches 4h Camarilla pivot point (mean reversion to equilibrium)
# Uses Camarilla levels from 4h for precise intraday structure, 1d EMA34 for higher timeframe trend filter
# Volume spike confirms institutional participation, reducing false breakouts
# Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
# Timeframe: 4h (primary timeframe as required)
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (based on previous bar's range)
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # Pivot = (high + low + close)/3
    # We use R3 and S3 as primary breakout levels, pivot as exit
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + 1.125 * camarilla_range
    camarilla_s3 = camarilla_pivot - 1.125 * camarilla_range
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation on 4h (threshold: 1.8x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.8 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND 1d EMA34 > price (uptrend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                ema_34_1d_aligned[i] > close[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 AND 1d EMA34 < price (downtrend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  ema_34_1d_aligned[i] < close[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches Camarilla pivot (mean reversion)
            if close[i] >= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches Camarilla pivot (mean reversion)
            if close[i] <= camarilla_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals