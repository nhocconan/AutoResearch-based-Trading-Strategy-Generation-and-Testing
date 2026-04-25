#!/usr/bin/env python3
"""
1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Camarilla H3/L3 breakouts in direction of weekly trend (EMA34) with volume confirmation.
Camarilla H3/L3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
Weekly EMA34 filter ensures trades align with higher-timeframe momentum, working in both bull and bear markets.
Volume spike confirms institutional participation. Designed for low trade frequency (7-25/year) to minimize fee drag.
Uses 1d primary timeframe with 1w HTF for trend calculation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla H3 and L3 levels (stronger breakout levels)
    camarilla_h3 = typical_price_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = typical_price_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA34 (34) and ensure Camarilla data is ready
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * average of last 20 periods
        if i >= 20:
            vol_avg = np.mean(volume[i-20:i])
            volume_spike = volume[i] > 2.0 * vol_avg
        else:
            volume_spike = False
        
        if position == 0:
            # Long: price breaks above H3 AND 1w trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > camarilla_h3_aligned[i]) and \
                         (close[i] > ema_34_1w_aligned[i]) and \
                         volume_spike
            # Short: price breaks below L3 AND 1w trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < camarilla_l3_aligned[i]) and \
                          (close[i] < ema_34_1w_aligned[i]) and \
                          volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters between L3 and H3 OR 1w trend turns bearish
            if (camarilla_l3_aligned[i] < close[i] < camarilla_h3_aligned[i]) or \
               (close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between L3 and H3 OR 1w trend turns bullish
            if (camarilla_l3_aligned[i] < close[i] < camarilla_h3_aligned[i]) or \
               (close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0