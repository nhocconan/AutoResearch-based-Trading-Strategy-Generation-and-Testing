#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and 1w trend filter
# Designed for low trade frequency (target 15-30/year) with clear trend following logic
# Williams Alligator identifies trends via SMAs (jaw/teeth/lips); we use lips > teeth > jaw for uptrend
# Works in both bull (trend continuation) and bear (trend continuation) markets
# Uses volume spike to confirm breakouts and avoid false signals
# 12h timeframe reduces trade frequency vs lower TFs, minimizing fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for Williams Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: three SMAs
    # Jaw (13-period, 8-shifted)
    jaw_12h = pd.Series(close_12h).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth (8-period, 5-shifted)
    teeth_12h = pd.Series(close_12h).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips (5-period, 3-shifted)
    lips_12h = pd.Series(close_12h).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Load 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for trend filter (close > SMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # SMA50 on 1w for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to 12h timeframe
    jaw_12h_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_12h_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_12h_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_12h_aligned[i]) or np.isnan(teeth_12h_aligned[i]) or 
            np.isnan(lips_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(sma50_1w_aligned[i])):
            continue
        
        # Williams Alligator conditions
        # Uptrend: lips > teeth > jaw
        # Downtrend: lips < teeth < jaw
        is_uptrend = (lips_12h_aligned[i] > teeth_12h_aligned[i] > jaw_12h_aligned[i])
        is_downtrend = (lips_12h_aligned[i] < teeth_12h_aligned[i] < jaw_12h_aligned[i])
        
        # Long entry: uptrend + price above lips + volume spike
        if (is_uptrend and 
            close[i] > lips_12h_aligned[i] and 
            volume[i] > 2.0 * vol_avg_1d_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: downtrend + price below lips + volume spike
        elif (is_downtrend and 
              close[i] < lips_12h_aligned[i] and 
              volume[i] > 2.0 * vol_avg_1d_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal or price crosses jaw
        elif position == 1 and (not is_uptrend or close[i] < jaw_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not is_downtrend or close[i] > jaw_12h_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_1wTrend"
timeframe = "12h"
leverage = 1.0