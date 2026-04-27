# 6h_WilliamsAlligator_TrendFilter_v1
# Williams Alligator (13,8,5) with 1d trend filter and volume confirmation
# Trend filter: price > 1d EMA50 for long, < 1d EMA50 for short
# Entry: price crosses above/below Alligator lips (13-period smoothed median)
# Exit: price crosses opposite Alligator teeth (8-period smoothed median)
# Volume filter: current volume > 1.5x 20-period average volume
# Designed for 6h timeframe with ~15-35 trades/year target

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator components (6h timeframe)
    # Jaw (13-period, smoothed 8 bars ahead)
    median_price = (high + low) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)  # Shift forward 8 bars
    jaw[:8] = jaw_raw[8] if len(jaw_raw) > 8 else 0  # Fill beginning
    
    # Teeth (8-period, smoothed 5 bars ahead)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)  # Shift forward 5 bars
    teeth[:5] = teeth_raw[5] if len(teeth_raw) > 5 else 0  # Fill beginning
    
    # Lips (5-period, smoothed 3 bars ahead)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)  # Shift forward 3 bars
    lips[:3] = lips_raw[3] if len(lips_raw) > 3 else 0  # Fill beginning
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup
    start_idx = max(50, 20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips[i]) or 
            np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema50_1d_aligned[i]
        lips_val = lips[i]
        teeth_val = teeth[i]
        jaw_val = jaw[i]
        vol_filt = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above lips AND price > 1d EMA50 AND volume filter
            if (close[i] > lips_val and close[i-1] <= lips_val and 
                close[i] > ema_trend and vol_filt):
                signals[i] = size
                position = 1
            # Short: price crosses below lips AND price < 1d EMA50 AND volume filter
            elif (close[i] < lips_val and close[i-1] >= lips_val and 
                  close[i] < ema_trend and vol_filt):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth (8-period)
            if close[i] < teeth_val and close[i-1] >= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above teeth (8-period)
            if close[i] > teeth_val and close[i-1] <= teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WilliamsAlligator_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0