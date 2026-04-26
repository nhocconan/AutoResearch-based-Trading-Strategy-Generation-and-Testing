#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: On daily timeframe, trade Camarilla R1/S1 breakouts from prior day with weekly EMA20 trend filter and daily volume spike confirmation. Target 7-25 trades/year by requiring confluence of weekly trend alignment, volume spike (>2x avg volume), and price structure breakout. Designed to work in both bull and bear markets via trend filter and volume confirmation to avoid false breakouts in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior 1d)
    cam_high = pd.Series(df_1d['high'].values).shift(1).values
    cam_low = pd.Series(df_1d['low'].values).shift(1).values
    cam_close = pd.Series(df_1d['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Get 1w data for HTF trend (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike: current volume > 2x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(20) 1w (20), Camarilla (need 2 bars for shift), volume MA (20)
    start_idx = max(20, 2, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_20_1w_val = ema_20_1w_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA20 (uptrend) or < EMA20 (downtrend)
        uptrend = close_val > ema_20_1w_val
        downtrend = close_val < ema_20_1w_val
        
        if position == 0:
            # Long: break above R1 with uptrend and volume spike
            long_signal = (close_val > r1_val) and \
                          uptrend and \
                          vol_spike
            
            # Short: break below S1 with downtrend and volume spike
            short_signal = (close_val < s1_val) and \
                           downtrend and \
                           vol_spike
            
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
            # Exit: close below Camarilla S1 (reversion to mean) or opposite breakout
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above Camarilla R1 (reversion to mean) or opposite breakout
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0