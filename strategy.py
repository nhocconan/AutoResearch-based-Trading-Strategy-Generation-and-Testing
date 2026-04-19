#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5) + 12h Trend Filter + Volume Confirmation
# Williams Alligator uses smoothed median prices (Jaw:13, Teeth:8, Lips:5) to identify trends
# When Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend, intertwined = sideways
# Trend filter: 12h EMA50 slope (rising/falling) to avoid counter-trend trades
# Volume: current > 1.5x 20-period average for confirmation
# Designed to catch strong trends while avoiding whipsaws in ranging markets
# Target: 15-25 trades/year per side to stay within 60-100 total/year for 6h
name = "6h_WilliamsAlligator_12hTrendFilter_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_slope = ema_50_12h - np.roll(ema_50_12h, 1)
    ema_50_12h_slope[0] = 0
    ema_50_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h_slope)
    
    # Williams Alligator components (using median price = (high+low)/2)
    median_price = (high + low) / 2
    
    # Jaw (13-period SMMA of median)
    jaw = pd.Series(median_price).ewm(alpha=1/13, adjust=False, min_periods=13).mean().values
    # Teeth (8-period SMMA of median)
    teeth = pd.Series(median_price).ewm(alpha=1/8, adjust=False, min_periods=8).mean().values
    # Lips (5-period SMMA of median)
    lips = pd.Series(median_price).ewm(alpha=1/5, adjust=False, min_periods=5).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema_50_12h_slope_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        # Williams Alligator signals
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Trend filter from 12h EMA50 slope
        uptrend_filter = ema_50_12h_slope_aligned[i] > 0
        downtrend_filter = ema_50_12h_slope_aligned[i] < 0
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 12h uptrend + volume
            if lips_above_teeth and teeth_above_jaw and uptrend_filter and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 12h downtrend + volume
            elif lips_below_teeth and teeth_below_jaw and downtrend_filter and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Lips cross below Teeth OR 12h trend turns down
            if lips[i] < teeth[i] or ema_50_12h_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Lips cross above Teeth OR 12h trend turns up
            if lips[i] > teeth[i] or ema_50_12h_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals