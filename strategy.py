#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 Breakout with 4h EMA20 Trend Filter and Volume Spike + Session Filter (08-20 UTC)
- Uses inner Camarilla levels (R3/S3) for more frequent but still high-probability breakouts
- 4h EMA20 defines short-term trend: only long when price > EMA20, short when price < EMA20
- Volume confirmation (> 1.8x 20-period average) filters weak breakouts
- Session filter (08-20 UTC) avoids low-liquidity Asian session noise
- Designed for 1h timeframe targeting 15-30 trades/year (60-120 over 4 years)
- Inner levels (R3/S3) provide better balance between frequency and reliability vs outer R4/S4
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1d Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 = C ± (H-L)*1.1/4 (inner levels)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align to 1h timeframe (use previous day's levels for breakout)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 20, 20)  # need 1d pivots, 4h EMA20, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 4h EMA20 AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_20_4h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND below 4h EMA20 AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_20_4h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: price returns to Camarilla H4/L4 levels OR crosses 4h EMA20
            exit_signal = False
            # Calculate H4/L4 for exit (outer levels)
            camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
            camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
            camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
            camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
            
            if position == 1:
                # Exit long when price < H4 OR < 4h EMA20
                if close[i] < camarilla_h4_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > L4 OR > 4h EMA20
                if close[i] > camarilla_l4_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA20_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0