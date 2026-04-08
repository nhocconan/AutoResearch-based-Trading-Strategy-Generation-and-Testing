#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elliott_wave_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Elliott Wave structure and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    n1w = len(close_1w)
    
    # Wave identification: identify swing points using 3-bar pivot (more responsive than fractals)
    # Bearish swing point: high[n] > high[n-1] and high[n] > high[n+1]
    # Bullish swing point: low[n] < low[n-1] and low[n] < low[n+1]
    swing_high = np.zeros(n1w, dtype=bool)
    swing_low = np.zeros(n1w, dtype=bool)
    
    for i in range(1, n1w - 1):
        if high_1w[i] > high_1w[i-1] and high_1w[i] > high_1w[i+1]:
            swing_high[i] = True
        if low_1w[i] < low_1w[i-1] and low_1w[i] < low_1w[i+1]:
            swing_low[i] = True
    
    # Elliott Wave trend: 5-wave impulse structure detection
    # Simplified: bullish trend when we have higher highs and higher lows
    # Bearish trend when we have lower highs and lower lows
    
    # Get last two swing points to determine trend
    last_swing_high_idx = np.where(swing_high)[0]
    last_swing_low_idx = np.where(swing_low)[0]
    
    # Arrays to store trend direction
    bullish_trend_1w = np.zeros(n1w, dtype=bool)
    bearish_trend_1w = np.zeros(n1w, dtype=bool)
    
    # Initialize with neutral
    bullish_trend_1w[:] = False
    bearish_trend_1w[:] = False
    
    # Determine trend based on recent swing points
    for i in range(2, n1w):
        # Get most recent swing high and low before current index
        recent_high_idx = last_swing_high_idx[last_swing_high_idx < i]
        recent_low_idx = last_swing_low_idx[last_swing_low_idx < i]
        
        if len(recent_high_idx) >= 2 and len(recent_low_idx) >= 2:
            hh1 = high_1w[recent_high_idx[-1]]
            hh2 = high_1w[recent_high_idx[-2]]
            ll1 = low_1w[recent_low_idx[-1]]
            ll2 = low_1w[recent_low_idx[-2]]
            
            # Bullish trend: higher highs and higher lows
            if hh1 > hh2 and ll1 > ll2:
                bullish_trend_1w[i] = True
                bearish_trend_1w[i] = False
            # Bearish trend: lower highs and lower lows
            elif hh1 < hh2 and ll1 < ll2:
                bearish_trend_1w[i] = True
                bullish_trend_1w[i] = False
            # Equal or mixed: no clear trend
            else:
                bullish_trend_1w[i] = False
                bearish_trend_1w[i] = False
        elif len(recent_high_idx) >= 1 and len(recent_low_idx) >= 1:
            # Not enough points for HH/LL comparison, use single point
            bullish_trend_1w[i] = close_1w[i] > np.mean([high_1w[recent_high_idx[-1]], low_1w[recent_low_idx[-1]]])
            bearish_trend_1w[i] = close_1w[i] < np.mean([high_1w[recent_high_idx[-1]], low_1w[recent_low_idx[-1]]])
        else:
            bullish_trend_1w[i] = False
            bearish_trend_1w[i] = False
    
    # Align Elliott Wave trend to 6h timeframe
    bullish_trend_6w = align_htf_to_ltf(prices, df_1w, bullish_trend_1w.astype(float))
    bearish_trend_6w = align_htf_to_ltf(prices, df_1w, bearish_trend_1w.astype(float))
    
    # Volume filter: 6h volume > 1.3x 20-period average (slightly relaxed for more signals)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(bullish_trend_6w[i]) or np.isnan(bearish_trend_6w[i]) or 
            np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish trend emerges or volume dries up
            if bearish_trend_6w[i] == 1.0 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish trend emerges or volume dries up
            if bullish_trend_6w[i] == 1.0 or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter from Elliott Wave structure
            bullish = bullish_trend_6w[i] == 1.0
            bearish = bearish_trend_6w[i] == 1.0
            
            # Long: bullish Elliott Wave trend + volume
            if bullish and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short: bearish Elliott Wave trend + volume
            elif bearish and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals