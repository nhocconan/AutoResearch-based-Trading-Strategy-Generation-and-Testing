#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation
# Designed to capture institutional-level breakouts aligned with daily trend while filtering false moves.
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Works in bull/bear markets by following 1d EMA34 direction and requiring volume confirmation for validity.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla levels and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from completed 1d bars
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * camarilla_range / 2
    camarilla_s3 = close_1d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid EMA and volume EMA
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + above 1d EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and price_above_ema and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Camarilla S3 + below 1d EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and price_below_ema and volume_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses 1d trend alignment
            if close[i] < camarilla_s3_aligned[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses 1d trend alignment
            if close[i] > camarilla_r3_aligned[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals