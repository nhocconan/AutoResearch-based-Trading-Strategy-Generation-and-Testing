#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA(50) trend filter and volume confirmation
# Camarilla pivots provide statistically significant support/resistance levels from prior day
# R3/S3 levels act as strong reversal points; breaks suggest continuation with institutional participation
# 12h EMA(50) ensures alignment with intermediate trend to avoid counter-trend trades
# Volume spike (>1.8x 24-period EMA) confirms breakout validity
# Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag

name = "6h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla formulas (based on prior bar)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_high = close_12h + 1.1 * (high_12h - low_12h)  # R3
    camarilla_low = close_12h - 1.1 * (high_12h - low_12h)   # S3
    
    # Align Camarilla levels to 6h timeframe (wait for completed 12h bar)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 24-period EMA on 6h volume (4 * 6h = 24h ~ 1d)
    vol_series = pd.Series(volume)
    vol_ema_24 = vol_series.ewm(span=24, adjust=False, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 24-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_24[i])
        
        # Camarilla R3/S3 breakout signals with 1d trend filter
        # Long: Break above Camarilla R3 + price above 1d EMA50 + volume spike
        # Short: Break below Camarilla S3 + price below 1d EMA50 + volume spike
        if position == 0:
            if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversion to mean) OR below 1d EMA50
            if close[i] < camarilla_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversion to mean) OR above 1d EMA50
            if close[i] > camarilla_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals