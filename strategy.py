#!/usr/bin/env python3
"""
6h_Camarilla_R1S1_Breakout_1dTrend_Volume_v1
Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume confirmation.
Long when price breaks above R1 in 1d uptrend with volume spike.
Short when price breaks below S1 in 1d downtrend with volume spike.
Uses discrete sizing (0.25) to limit fee churn. Target: 50-150 trades over 4 years.
Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend) via regime filter.
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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of completed daily bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    # R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6), 
    # S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + (rng * 1.1 / 12)
    camarilla_s1 = close_1d - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d trend filter: EMA(34) on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(34) period, volume MA(20)
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        regime_long = close_val > ema_34_1d_aligned[i]  # 1d uptrend
        regime_short = close_val < ema_34_1d_aligned[i]  # 1d downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 1d uptrend
            long_signal = (close_val > camarilla_r1_aligned[i]) and vol_conf and regime_long
            
            # Short: price breaks below S1 AND volume confirm AND 1d downtrend
            short_signal = (close_val < camarilla_s1_aligned[i]) and vol_conf and regime_short
            
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
            # Exit: price crosses below Camarilla pivot point (close_1d) OR 1d trend flips down
            pivot_point = (high_1d[i] + low_1d[i] + close_1d[i]) / 3  # approximate PP
            # Align pivot point to 6h
            pp_aligned = align_htf_to_ltf(prices, df_1d, 
                                        np.full_like(close_1d, 
                                                    (high_1d + low_1d + close_1d) / 3))
            if (close_val < pp_aligned[i]) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price crosses above Camarilla pivot point OR 1d trend flips up
            pp_aligned = align_htf_to_ltf(prices, df_1d, 
                                        np.full_like(close_1d, 
                                                    (high_1d + low_1d + close_1d) / 3))
            if (close_val > pp_aligned[i]) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0