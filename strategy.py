#!/usr/bin/env python3
"""
4h Williams Alligator + 1d EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (SMAs with specific periods) identifies trend direction and strength.
When price is aligned with the Alligator (above lips/teeth/jaw for long, below for short) and 
confirmed by 1d EMA34 trend and volume spike, it captures strong institutional moves.
Works in bull/bear markets via trend filter. Target: 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
    # Using SMAs as per original Alligator
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator calculation (max shift 8 + window 13 = 21) + EMA34 warmup
    start_idx = max(34, 21)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Alligator conditions
        # Long: price above all three lines (lips > teeth > jaw) AND above 1d EMA34
        long_alligator = (lips[i] > teeth[i]) and (teeth[i] > jaw[i]) and (curr_close > lips[i])
        # Short: price below all three lines (jaw > teeth > lips) AND below 1d EMA34
        short_alligator = (jaw[i] > teeth[i]) and (teeth[i] > lips[i]) and (curr_close < jaw[i])
        
        # Entry signals with trend and volume filters
        if position == 0:
            long_condition = long_alligator and (curr_close > ema_trend) and volume_spike
            short_condition = short_alligator and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below teeth or trend breaks
            if curr_close < teeth[i] or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above teeth or trend breaks
            if curr_close > teeth[i] or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0