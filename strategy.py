#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA50 trend + volume confirmation.
# Long when Alligator jaws (13-period smoothed median) turns up and price > teeth (8-period),
# with 1d uptrend (close > 1d EMA50) and volume > 1.8x 20-bar avg.
# Short when jaws turn down and price < teeth, with 1d downtrend (close < 1d EMA50) and volume spike.
# Exit on Alligator reversal (jaws cross teeth) or opposite signal.
# Uses Williams Alligator for trend detection with smoothness to reduce whipsaws,
# 1d EMA50 for stronger trend filter, and volume confirmation to avoid low-quality breakouts.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Williams Alligator components (using median price)
    median_price = (high + low) / 2
    
    # Jaws: 13-period SMMA of median, shifted 8 bars
    jaws = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaws = np.roll(jaws, 8)
    jaws[:8] = np.nan
    
    # Teeth: 8-period SMMA of median, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMMA of median, shifted 3 bars (not used in signals but for completeness)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaws[i]) or np.isnan(teeth[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_jaws = jaws[i]
        curr_teeth = teeth[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: jaws > teeth (uptrend), price > teeth, 1d uptrend, volume spike
            if (curr_jaws > curr_teeth and 
                curr_close > curr_teeth and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: jaws < teeth (downtrend), price < teeth, 1d downtrend, volume spike
            elif (curr_jaws < curr_teeth and 
                  curr_close < curr_teeth and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: jaws < teeth (trend reversal to down)
            if curr_jaws < curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: jaws > teeth (trend reversal to up)
            if curr_jaws > curr_teeth:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals