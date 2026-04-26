#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1
Hypothesis: On 4h timeframe, trade Camarilla R1/S1 breakouts from prior 4h bar with 1d EMA34 trend filter and volume spike confirmation. Target 20-50 trades/year by requiring confluence of daily trend alignment, above-average volume, and price structure breakout. Designed to work in both bull and bear markets via trend filter and volume confirmation to avoid false breakouts in chop.
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
    
    # Get 4h data for Camarilla levels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for HTF trend (EMA34) and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d average volume for volume spike filter (20-period SMA)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior 4h)
    cam_high = pd.Series(df_4h['high'].values).shift(1).values
    cam_low = pd.Series(df_4h['low'].values).shift(1).values
    cam_close = pd.Series(df_4h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Align HTF indicators to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(34) 1d, volume avg (20), Camarilla (need 2 bars for shift)
    start_idx = max(34, 20, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        vol_avg_1d_val = vol_avg_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_val = volume[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend) or < EMA34 (downtrend)
        uptrend = close_val > ema_34_1d_val
        downtrend = close_val < ema_34_1d_val
        
        # Volume filter: current volume > 1.5x daily average volume
        volume_filter = vol_val > 1.5 * vol_avg_1d_val
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            long_signal = (close_val > r1_val) and \
                          uptrend and \
                          volume_filter
            
            # Short: break below S1 with downtrend and volume spike
            short_signal = (close_val < s1_val) and \
                           downtrend and \
                           volume_filter
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below S1 (reversal signal) or close below EMA34 (trend change)
            if close_val < s1_val or close_val < ema_34_1d_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above R1 (reversal signal) or close above EMA34 (trend change)
            if close_val > r1_val or close_val > ema_34_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0