#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Camarilla pivots from 12h timeframe provide institutional support/resistance levels
# Breakout above R3 or below S3 with volume confirmation indicates strong institutional participation
# 12h EMA50 filter ensures we only trade in direction of higher timeframe trend
# Volume spike (>2.0x 20-period EMA) confirms breakout validity
# Works in both bull and bear markets by following the 12h trend direction
# Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag

name = "6h_Camarilla_R3S3_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla pivot levels (R3, S3)
    # Camarilla: R4 = close + ((high - low) * 1.1/2), R3 = close + ((high - low) * 1.1/4)
    #          S3 = close - ((high - low) * 1.1/4), S4 = close - ((high - low) * 1.1/2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot levels for each 12h bar
    camarilla_r3 = close_12h + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3 = close_12h - ((high_12h - low_12h) * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe (with 1-bar delay for completed 12h bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla R3/S3 breakout with 12h trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above R3 + above 12h EMA50 + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + below 12h EMA50 + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and volume_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below R3 OR below 12h EMA50
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above S3 OR above 12h EMA50
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals