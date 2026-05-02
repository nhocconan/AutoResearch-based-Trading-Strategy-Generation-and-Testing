#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with 1d EMA50 trend filter and volume spike
# Williams Alligator (jaw=13, teeth=8, lips=5 SMAs) identifies trend when lines are aligned and separated
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume spike confirmation (>2.0 x 20-period EMA) filters false breakouts
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull markets (price > Alligator lines + 1d EMA50 up) and bear markets (price < Alligator lines + 1d EMA50 down)

name = "12h_WilliamsAlligator_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator components (13,8,5 SMAs on median price)
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw (13-period SMA)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator calculation)
    start_idx = 13
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator alignment (trending condition)
        # Alligator is "awake" when lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
        alligator_uptrend = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_downtrend = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price above Alligator lines with volume confirmation and aligned uptrend
            if close[i] > lips[i] and close[i] > teeth[i] and close[i] > jaw[i] and \
               volume_confirmation[i] and alligator_uptrend and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Price below Alligator lines with volume confirmation and aligned downtrend
            elif close[i] < lips[i] and close[i] < teeth[i] and close[i] < jaw[i] and \
                 volume_confirmation[i] and alligator_downtrend and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Alligator teeth (weakening trend) OR trend changes to downtrend
            if close[i] < teeth[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Alligator teeth (weakening trend) OR trend changes to uptrend
            if close[i] > teeth[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals