#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike (>2x median). 
Targets intraday momentum continuation within higher timeframe trend. Uses volume filter to avoid low-conviction breakouts.
Designed for both bull/bear markets by aligning with 4h trend and requiring volume confirmation.
Target: 15-35 trades/year via tight entry conditions (breakout + trend + volume).
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
    
    # Get 4h data for HTF trend (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume filter (median volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d median volume (20-period) for volume spike filter
    vol_median_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).median().values
    
    # Calculate Camarilla levels from previous 1h bar (OHLC of prior 1h)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    cam_high = pd.Series(df_1h['high'].values).shift(1).values
    cam_low = pd.Series(df_1h['low'].values).shift(1).values
    cam_close = pd.Series(df_1h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (primary breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 4h, volume median (20), Camarilla (need 2 bars for shift)
    start_idx = max(50, 20, 2) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_median_1d_aligned[i]) or
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_4h_val
        downtrend = close_val < ema_50_4h_val
        
        # Volume spike filter: volume > 2x median volume for conviction
        volume_spike = volume_val > 2.0 * vol_median_1d_val
        
        if position == 0:
            # Long: break above R1 with volume spike, and uptrend
            long_signal = (close_val > r1_val) and \
                          volume_spike and \
                          uptrend
            
            # Short: break below S1 with volume spike, and downtrend
            short_signal = (close_val < s1_val) and \
                           volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit on close below S1 (mean reversion) or trend change
            if close_val < s1_val or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit on close above R1 (mean reversion) or trend change
            if close_val > r1_val or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolumeFilter_v1"
timeframe = "1h"
leverage = 1.0