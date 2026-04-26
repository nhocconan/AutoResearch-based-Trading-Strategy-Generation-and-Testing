#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, trade Camarilla R3/S3 breakouts with 1d trend filter and volume confirmation. 
R3/S3 represent stronger reversal/continuation levels than R1/S1. In 2022-2024, these levels captured 
major swing points during volatile periods. Volume confirmation reduces false breakouts. 
Discrete sizing (0.0, ±0.25) minimizes fee churn. Targets 12-37 trades per year over 4 years.
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
    
    # Get 1d data for HTF trend filter (EMA34) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # where C = (H+L+C)/3 (typical price)
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_R3_1d = typical_price_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_S3_1d = typical_price_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_R3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    # Calculate volume average (20-period) for confirmation
    volume_s = pd.Series(volume)
    volume_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup + volume MA warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R3_1d_aligned[i]) or 
            np.isnan(camarilla_S3_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_34_1d_aligned[i]
        trend_downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume confirm + 1d uptrend
            long_signal = (close[i] > camarilla_R3_1d_aligned[i] and 
                          volume_confirm and 
                          trend_uptrend)
            
            # Short: price breaks below S3 + volume confirm + 1d downtrend
            short_signal = (close[i] < camarilla_S3_1d_aligned[i] and 
                           volume_confirm and 
                           trend_downtrend)
            
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
            # Exit: price breaks below S3 (mean reversion) OR trend change to downtrend
            if close[i] < camarilla_S3_1d_aligned[i] or not trend_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 (mean reversion) OR trend change to uptrend
            if close[i] > camarilla_R3_1d_aligned[i] or not trend_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0