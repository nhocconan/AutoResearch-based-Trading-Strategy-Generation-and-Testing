#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike
Hypothesis: Trade 4h Camarilla R1/S1 breakouts in direction of 12h trend (EMA50) with volume confirmation.
Camarilla levels provide high-probability breakout points. Trend filter avoids counter-trend trades.
Volume spike confirms institutional participation. Designed for low trade frequency (20-50/year) to minimize fee drag.
Uses 4h primary timeframe with 12h HTF for trend and 1d for Camarilla calculation.
Works in bull markets (breakouts with trend) and bear markets (breakouts with trend, short bias).
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
    
    # Get 1d data for Camarilla levels (calculated from daily candles)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 1d bar
    # Typical price = (H + L + C) / 3
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla R1 and S1 levels (main breakout levels)
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h EMA50 (50) and ensure Camarilla data is ready
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * average of last 20 periods
        if i >= 20:
            vol_avg = np.mean(volume[i-20:i])
            volume_spike = volume[i] > 2.0 * vol_avg
        else:
            volume_spike = False
        
        if position == 0:
            # Long: price breaks above R1 AND 12h trend bullish (close > EMA50) AND volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema_50_12h_aligned[i]) and \
                         volume_spike
            # Short: price breaks below S1 AND 12h trend bearish (close < EMA50) AND volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema_50_12h_aligned[i]) and \
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
            # Exit: price re-enters between S1 and R1 OR 12h trend turns bearish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between S1 and R1 OR 12h trend turns bullish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrendFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0