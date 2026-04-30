#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment), price > 1d EMA50, and volume > 1.8x 20-bar avg.
# Short when Lips < Teeth < Jaw (bearish alignment), price < 1d EMA50, and volume > 1.8x 20-bar avg.
# Exit when Alligator alignment breaks (mean reversion).
# Williams Alligator identifies trend absence/presence via smoothed medians, effective in ranging and trending markets.
# Combined with 1d EMA50 trend filter to avoid counter-trend trades and volume confirmation to reduce false signals.
# Timeframe: 12h as per experiment guidelines.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all smoothed with offset
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Lips > Teeth > Jaw (bullish), price > 1d EMA50, volume spike
            if (curr_lips > curr_teeth and 
                curr_teeth > curr_jaw and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish), price < 1d EMA50, volume spike
            elif (curr_lips < curr_teeth and 
                  curr_teeth < curr_jaw and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if (curr_lips <= curr_teeth or 
                curr_teeth <= curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if (curr_lips >= curr_teeth or 
                curr_teeth >= curr_jaw):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals