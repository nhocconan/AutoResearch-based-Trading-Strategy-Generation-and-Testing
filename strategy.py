#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data once for Camarilla and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Using previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels
    r3 = pivot_1d + (range_1d * 1.1 / 4)
    s3 = pivot_1d - (range_1d * 1.1 / 4)
    r4 = pivot_1d + (range_1d * 1.1 / 2)
    s4 = pivot_1d - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d trend filter: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume spike and uptrend
            long_breakout = (close[i] > r4_aligned[i] and 
                           ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                           volume_spike[i])
            
            # Short breakout: price breaks below S4 with volume spike and downtrend
            short_breakout = (close[i] < s4_aligned[i] and 
                            ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                            volume_spike[i])
            
            # Fade at R3/S3 in ranging conditions (when trend is weak)
            # Only fade if price is near extreme and trend is not strong
            trend_strength = abs(ema_34_1d_aligned[i] - ema_34_1d_aligned[i-20]) / close[i]
            is_ranging = trend_strength < 0.01  # Less than 1% change over 20 periods
            
            fade_long = (close[i] < s3_aligned[i] and 
                        is_ranging and
                        volume_spike[i])
            
            fade_short = (close[i] > r3_aligned[i] and 
                         is_ranging and
                         volume_spike[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            elif fade_long:
                signals[i] = 0.25
                position = 1
            elif fade_short:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below R3 or trend reverses
            if (close[i] < r3_aligned[i] or 
                ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above S3 or trend reverses
            if (close[i] > s3_aligned[i] or 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals