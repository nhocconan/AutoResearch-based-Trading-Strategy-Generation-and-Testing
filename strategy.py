#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips (bullish alignment) AND 1w close > 1w EMA50 AND volume > 2.0x 20 EMA
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips (bearish alignment) AND 1w close < 1w EMA50 AND volume > 2.0x 20 EMA
# Uses discrete sizing (0.30) to limit fee drag. Target: 15-25 trades/year per symbol.
# Williams Alligator identifies trendless markets via intertwined lines; strong trends show clear separation.
# 1w EMA50 filters counter-trend trades; volume confirms momentum strength.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # Need at least 13 for Alligator (8,5,3)
        return np.zeros(n)
    
    # Calculate Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA but different smoothing)
    # Using EWA with alpha=1/period as approximation for SMMA
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    median_1d = (high_1d + low_1d) / 2  # Typical price for Alligator
    
    # Calculate SMMA using EWM with alpha=1/period (approximation)
    jaw_1d = pd.Series(median_1d).ewm(alpha=1/13, adjust=False).mean().values
    teeth_1d = pd.Series(median_1d).ewm(alpha=1/8, adjust=False).mean().values
    lips_1d = pd.Series(median_1d).ewm(alpha=1/5, adjust=False).mean().values
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_1d = np.roll(jaw_1d, 8)
    teeth_1d = np.roll(teeth_1d, 5)
    lips_1d = np.roll(lips_1d, 3)
    
    # Handle NaN from rolls
    jaw_1d[:8] = np.nan
    teeth_1d[:5] = np.nan
    lips_1d[:3] = np.nan
    
    # Bullish alignment: Lips > Teeth > Jaw
    # Bearish alignment: Lips < Teeth < Jaw
    bullish_align = (lips_1d > teeth_1d) & (teeth_1d > jaw_1d)
    bearish_align = (lips_1d < teeth_1d) & (teeth_1d < jaw_1d)
    
    # Align Alligator components to 1d timeframe (no additional delay needed as we use completed 1d bar)
    jaw_1d_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_1d_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_1d_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bullish_align_aligned = align_htf_to_ltf(prices, df_1d, bullish_align.astype(float))
    bearish_align_aligned = align_htf_to_ltf(prices, df_1d, bearish_align.astype(float))
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Uptrend when close > EMA50, downtrend when close < EMA50
    uptrend_1w = close_1w > ema_50_1w
    downtrend_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w.astype(float))
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_1d_aligned[i]) or np.isnan(teeth_1d_aligned[i]) or 
            np.isnan(lips_1d_aligned[i]) or np.isnan(bullish_align_aligned[i]) or 
            np.isnan(bearish_align_aligned[i]) or np.isnan(uptrend_1w_aligned[i]) or 
            np.isnan(downtrend_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bullish alignment AND 1w uptrend AND volume spike
            if (bullish_align_aligned[i] > 0.5 and 
                uptrend_1w_aligned[i] > 0.5 and 
                volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: Bearish alignment AND 1w downtrend AND volume spike
            elif (bearish_align_aligned[i] > 0.5 and 
                  downtrend_1w_aligned[i] > 0.5 and 
                  volume_spike[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Bearish alignment OR 1w trend changes to downtrend
            if (bearish_align_aligned[i] > 0.5 or 
                downtrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Bullish alignment OR 1w trend changes to uptrend
            if (bullish_align_aligned[i] > 0.5 or 
                uptrend_1w_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals