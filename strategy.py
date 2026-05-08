#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Trend Filter and Volume Spike
# - Williams Alligator (Jaw/Teeth/Lips) identifies trend absence/presence
# - Trade when Lips cross above/below Teeth with 1d trend alignment and volume spike
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 12-37 trades/year to minimize fee drag on 12h timeframe

name = "12h_Williams_Alligator_1dTrend_Volume"
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
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs with specific periods and shifts
    # Jaw: 13-period SMA, shifted 8 bars forward
    # Teeth: 8-period SMA, shifted 5 bars forward
    # Lips: 5-period SMA, shifted 3 bars forward
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips cross above Teeth + 1d uptrend + volume spike
            long_cond = (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and
                        ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Lips cross below Teeth + 1d downtrend + volume spike
            short_cond = (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and
                         ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Lips cross below Teeth (trend weakness)
            if lips[i] < teeth[i] and lips[i-1] >= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Lips cross above Teeth (trend weakness)
            if lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals