#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume spike confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long: Bull Power > 0 AND price > 1d EMA50 AND volume spike (>2.0x 20-bar average)
# Short: Bear Power < 0 AND price < 1d EMA50 AND volume spike
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe
# Works in bull via trend continuation, in bear via mean reversion at extremes

name = "6h_ElderRay_1dEMA50_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # max(13, 50) warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (weakening momentum) OR price below 1d EMA50 (trend change)
            if curr_bull <= 0 or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 (weakening momentum) OR price above 1d EMA50 (trend change)
            if curr_bear >= 0 or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (strong momentum) AND price above 1d EMA50 AND volume spike
            if (curr_bull > 0 and 
                curr_close > curr_ema_1d and
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 (strong momentum) AND price below 1d EMA50 AND volume spike
            elif (curr_bear < 0 and 
                  curr_close < curr_ema_1d and
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals