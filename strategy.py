#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA(50) trend filter and volume confirmation
# Designed for 12h timeframe to capture multi-day institutional breakouts aligned with weekly trend.
# Uses weekly EMA50 for strong trend filter to avoid counter-trend trades in choppy markets.
# Volume confirmation requires 2.0x 20-period volume EMA to ensure genuine participation.
# Discrete position sizing (0.25) balances profit potential and drawdown control.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_VolumeSpike"
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
    
    # Get 1w data for Camarilla levels and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from completed 1w bars
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_range = high_1w - low_1w
    camarilla_r3 = close_1w + 1.1 * camarilla_range / 2
    camarilla_s3 = close_1w - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1w bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid EMA and volume EMA
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 1w EMA50 + volume spike
            if close[i] > camarilla_r3_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + below 1w EMA50 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 1w trend alignment
            if close[i] < camarilla_s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 1w trend alignment
            if close[i] > camarilla_r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals