#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
# Long when Alligator jaws < teeth < lips (bullish alignment), price > lips, close > 1d EMA50, volume > 1.5x 20-bar avg.
# Short when Alligator jaws > teeth > lips (bearish alignment), price < lips, close < 1d EMA50, volume > 1.5x 20-bar avg.
# Exit when Alligator alignment reverses or price crosses lips in opposite direction.
# Uses 12h timeframe for low trade frequency (target: 12-37/year) to minimize fee drag.
# Williams Alligator identifies trend initiation and continuation with smoothed medians.
# 1d EMA50 filters for higher timeframe trend alignment.
# Volume confirmation reduces false signals.
# Works in bull markets via bullish Alligator alignment and in bear markets via bearish alignment.
# Target: 50-150 total trades over 4 years.

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe (using close prices)
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    # Lips: 5-period SMMA, shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().shift(5)
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().shift(3)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 70  # warmup for Alligator and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaw = jaw.iloc[i]
        curr_teeth = teeth.iloc[i]
        curr_lips = lips.iloc[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: jaw < teeth < lips
            bullish_align = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
            # Bearish alignment: jaw > teeth > lips
            bearish_align = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
            
            # Long: bullish alignment, price > lips, close > 1d EMA50, volume spike
            if (bullish_align and 
                curr_close > curr_lips and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment, price < lips, close < 1d EMA50, volume spike
            elif (bearish_align and 
                  curr_close < curr_lips and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: bearish alignment OR price crosses below lips
            bearish_align = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
            if bearish_align or (curr_close < curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: bullish alignment OR price crosses above lips
            bullish_align = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
            if bullish_align or (curr_close > curr_lips):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals