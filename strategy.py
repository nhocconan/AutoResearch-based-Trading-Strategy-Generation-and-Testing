#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Williams Alligator (JAW/TEETH/LIPS) identifies trend direction and strength.
# EMA50 on 1w filters for long-term trend alignment, reducing whipsaws.
# Volume > 1.5x 20-period average confirms institutional participation.
# Target: 10-30 trades over 4 years for low frequency, high conviction trades.
name = "1d_WilliamsAlligator_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d: SMAs of median price
    median_price = (df_1d['high'] + df_1d['low']) / 2
    jaw = median_price.rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = median_price.rolling(window=8, min_periods=8).mean().shift(5).values
    lips = median_price.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Williams Alligator to 1d (already aligned, but ensure proper shifting)
    jaw_1d = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_1d = align_htf_to_ltf(prices, df_1d, teeth)
    lips_1d = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Alligator and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1d[i]) or np.isnan(teeth_1d[i]) or np.isnan(lips_1d[i]) or 
            np.isnan(ema_50_1d[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_alignment = lips_1d[i] > teeth_1d[i] and teeth_1d[i] > jaw_1d[i]
        bearish_alignment = lips_1d[i] < teeth_1d[i] and teeth_1d[i] < jaw_1d[i]
        
        trend_up = close[i] > ema_50_1d[i]
        trend_down = close[i] < ema_50_1d[i]
        
        if position == 0:
            # Long: bullish Alligator alignment + uptrend + volume confirmation
            if bullish_alignment and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment + downtrend + volume confirmation
            elif bearish_alignment and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish Alligator alignment or trend reversal
            if bearish_alignment or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish Alligator alignment or trend reversal
            if bullish_alignment or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals